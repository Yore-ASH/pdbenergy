"""Turn a molecular conformation into a graph the neural network can consume.

Representation choices
----------------------
A molecule is *not* a grid and *not* a sequence of fixed length: it is a set of
atoms with Cartesian coordinates.  The representation must therefore respect two
symmetries, or the model will waste capacity learning them:

* **Permutation invariance** - reordering the atoms must not change the energy.
  A sum/mean aggregation over atoms gives this for free.
* **Translation and rotation invariance** - rigid-body motion has zero energy.
  We never feed raw coordinates as such; every input feature is built from
  *interatomic distances*, which are invariant by construction.

Each sample becomes a graph ``G = (V, E)``:

* node ``i``  - atom ``i``, labelled by its atomic number ``Z`` (an embedding);
* edge ``ij`` - a pair of atoms closer than ``cutoff`` Angstrom, labelled by a
  Gaussian radial-basis expansion of the distance plus a covalent-bond flag.

The radius graph is what makes the model *local*: energy is a sum of local
chemical environments, so each atom only needs to see its neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import FeatureConfig

#: Element symbol -> atomic number.  Covers everything a protein contains plus
#: the common ions/metals that survive cleaning.
ELEMENT_TO_Z: dict[str, int] = {
    "H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "NE": 10, "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15, "S": 16, "CL": 17,
    "AR": 18, "K": 19, "CA": 20, "SC": 21, "TI": 22, "V": 23, "CR": 24, "MN": 25,
    "FE": 26, "CO": 27, "NI": 28, "CU": 29, "ZN": 30, "GA": 31, "GE": 32, "AS": 33,
    "SE": 34, "BR": 35, "KR": 36, "RB": 37, "SR": 38, "Y": 39, "ZR": 40, "MO": 42,
    "RU": 44, "RH": 45, "PD": 46, "AG": 47, "CD": 48, "IN": 49, "SN": 50, "SB": 51,
    "TE": 52, "I": 53, "XE": 54, "CS": 55, "BA": 56, "PT": 78, "AU": 79, "HG": 80,
    "PB": 82, "D": 1, "X": 0,
}

#: Number of embedding rows (Z = 0 reserved for "unknown atom").
MAX_Z = 119

#: Covalent radii in Angstrom, used for the bond flag.  Values follow the
#: standard Cordero et al. (2008) single-bond radii, rounded.
COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07, "S": 1.05,
    "CL": 1.02, "SE": 1.20, "BR": 1.20, "I": 1.39, "NA": 1.66, "MG": 1.41,
    "K": 2.03, "CA": 1.76, "ZN": 1.22, "FE": 1.32, "MN": 1.39, "CU": 1.32,
    "X": 0.80,
}


def atomic_numbers(elements: Sequence[str]) -> np.ndarray:
    """Map element symbols to atomic numbers (0 for unknown)."""
    return np.asarray(
        [ELEMENT_TO_Z.get(str(e).upper().strip(), 0) for e in elements], dtype=np.int64
    )


def gaussian_rbf(
    distances: np.ndarray,
    n_rbf: int,
    rbf_start: float,
    rbf_end: float,
) -> np.ndarray:
    """Expand distances into a smooth basis (Gaussian radial basis functions).

    A raw distance is a poor input for a neural network: it is unbounded, and
    the interesting structure (``1.45 A`` = covalent bond vs ``3.8 A`` =
    hydrogen bond) is squeezed into a tiny numeric range.  Expanding into
    overlapping Gaussians centred every ``(rbf_end - rbf_start) / n_rbf`` A gives
    the network a smooth, local, well-conditioned description of distance.
    """
    centers = np.linspace(rbf_start, rbf_end, n_rbf, dtype=np.float64)
    width = (rbf_end - rbf_start) / max(1, n_rbf - 1)
    gamma = 1.0 / (2.0 * width * width)
    d = np.asarray(distances, dtype=np.float64).reshape(-1, 1)
    return np.exp(-gamma * (d - centers.reshape(1, -1)) ** 2).astype(np.float32)


def cutoff_envelope(distances: np.ndarray, cutoff: float) -> np.ndarray:
    """Smoothly go to zero at the cutoff (a cosine window).

    A hard cutoff makes the energy discontinuous: an atom pair crossing the
    cutoff makes the prediction jump.  Multiplying the message by
    ``0.5 * (cos(pi r / rc) + 1)`` removes that discontinuity, which matters for
    smoothness-based applications such as forces, geometry optimisation and MD.
    """
    r = np.asarray(distances, dtype=np.float64)
    value = 0.5 * (np.cos(np.pi * np.clip(r, 0.0, cutoff) / cutoff) + 1.0)
    return np.where(r < cutoff, value, 0.0).astype(np.float32)


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Full (N, N) Euclidean distance matrix with an infinite diagonal.

    One O(N^2) computation, reused by both the radius graph and the bond flags.
    For N <= ~1500 atoms this costs a few milliseconds and is faster than
    building a cell list; it is the single hot spot of featurisation, which is
    why it is computed exactly once per frame.
    """
    coords = np.asarray(coords, dtype=np.float64)
    # (x - y)^2 = x^2 + y^2 - 2xy, evaluated as a BLAS matrix product.
    sq = np.einsum("ij,ij->i", coords, coords)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (coords @ coords.T)
    np.maximum(d2, 0.0, out=d2)
    dist = np.sqrt(d2)
    np.fill_diagonal(dist, np.inf)
    return dist


def _radius_graph(
    dist: np.ndarray, cutoff: float, max_neighbors: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Atom pairs closer than ``cutoff``, as (senders, receivers, distances)."""
    mask = dist < cutoff

    if max_neighbors and max_neighbors > 0:
        # Safety valve: keep only the closest `max_neighbors` partners per atom.
        if mask.sum(axis=1).max() > max_neighbors:
            # argpartition is O(N^2); a full argsort would be O(N^2 log N).
            kth = min(max_neighbors, dist.shape[1] - 1)
            nearest = np.argpartition(dist, kth, axis=1)[:, :max_neighbors]
            keep = np.zeros_like(mask)
            rows = np.arange(dist.shape[0])[:, None]
            keep[rows, nearest] = mask[rows, nearest]
            mask = keep

    senders, receivers = np.nonzero(mask)
    return senders, receivers, dist[senders, receivers]


def bond_flags_from_distances(dist: np.ndarray, elements: Sequence[str]) -> np.ndarray:
    """1.0 where a pair is covalently bonded, from a precomputed distance matrix.

    The force field's bonded terms (stretch, bend, torsion) are the largest
    terms in the energy, and which atoms are bonded is pure chemical graph
    information that the network would otherwise have to infer from distance
    alone.  A pair counts as bonded when
    ``d < 1.3 * (r_cov(i) + r_cov(j))``.
    """
    radii = np.asarray([COVALENT_RADII.get(str(e).upper(), 0.8) for e in elements])
    threshold = 1.3 * (radii[:, None] + radii[None, :])
    return (dist < threshold).astype(np.float32)


@dataclass
class GraphArrays:
    """A featurised conformation, still in NumPy (converted to tensors later)."""

    z: np.ndarray            # (N,) int64
    pos: np.ndarray          # (N, 3) float32
    edge_index: np.ndarray   # (2, E) int64
    edge_attr: np.ndarray    # (E, F) float32
    n_atoms: int
    n_edges: int


def frame_to_graph(
    coords: np.ndarray,
    elements: Sequence[str],
    cfg: FeatureConfig,
) -> GraphArrays:
    """Featurise one conformation (Angstrom coordinates) into graph arrays.

    Edge features are deliberately kept **minimal**: one channel for the
    interatomic distance and (optionally) one for the covalent-bond flag.  The
    radial-basis expansion happens inside the network
    (:class:`pdbenergy.models.GaussianRBFExpansion`) rather than here.

    This matters at scale.  Expanding every edge into 48 Gaussians here would
    store ~49 floats per edge; with ~20,000 edges per conformation and ~1300
    conformations that is gigabytes of mostly zeros - a Gaussian basis is
    *sparse*, only ~3-4 of the 48 functions are non-negligible at any given
    distance.  Keeping the raw distance shrinks the cached dataset ~25x and
    turns the expansion into one vectorised op inside the forward pass.
    """
    coords = np.asarray(coords, dtype=np.float64)
    z = atomic_numbers(elements)
    dist = pairwise_distances(coords)
    senders, receivers, edge_dist = _radius_graph(dist, cfg.cutoff, cfg.max_neighbors)

    # Edge direction convention: index 0 = receiver (updates), 1 = sender.
    edge_index = np.stack([receivers, senders], axis=0).astype(np.int64)

    columns = [edge_dist.astype(np.float32)]
    if cfg.use_bond_feature:
        flags = bond_flags_from_distances(dist, elements)
        columns.append(flags[receivers, senders])
    edge_attr = np.stack(columns, axis=1).astype(np.float32)

    return GraphArrays(
        z=z,
        pos=coords.astype(np.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        n_atoms=int(coords.shape[0]),
        n_edges=int(edge_index.shape[1]),
    )


# --------------------------------------------------------------------------- #
# Hand-crafted global descriptors (the baseline model's input)
# --------------------------------------------------------------------------- #


def global_descriptors(coords: np.ndarray, elements: Sequence[str]) -> np.ndarray:
    """A fixed-length, physics-motivated summary of one conformation.

    These are the kind of features a chemist would write down by hand: size,
    shape, composition, and crude counts of contacts and electrostatic energy.
    They are permutation- and rotation-invariant, but they *throw away* which
    atom sits next to which - which is exactly the information the graph model
    keeps.  Comparing the two models shows what that information is worth.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if n == 0:
        return np.zeros(24, dtype=np.float32)
    elements = [str(e).upper() for e in elements]

    centered = coords - coords.mean(axis=0, keepdims=True)
    rg = float(np.sqrt(np.mean(np.einsum("ij,ij->i", centered, centered))))
    # Gyration tensor eigenvalues: 3 shape numbers (rod / disc / sphere).
    gyration = centered.T @ centered / n
    eigvals = np.sort(np.linalg.eigvalsh(gyration))[::-1]
    anisotropy = float((eigvals[0] - eigvals[2]) / (eigvals.sum() + 1e-8))

    dist = pairwise_distances(coords)
    iu = np.triu_indices(n, k=1)
    pair_dist = dist[iu]
    max_dist = float(pair_dist.max()) if pair_dist.size else 0.0

    # Monopole electrostatic and LJ-like energies with unit generic parameters:
    # deliberately crude, but they capture real long-range physics.
    inv_r = 1.0 / np.clip(pair_dist, 1e-6, None)
    elec = float(np.sum(inv_r))
    sigma = 3.4
    eps = 0.1
    ratio6 = (sigma / np.clip(pair_dist, 1e-6, None)) ** 6
    lj = float(np.sum(4.0 * eps * (ratio6 ** 2 - ratio6)))

    z = atomic_numbers(elements)
    counts = np.array([(z == k).sum() for k in (1, 6, 7, 8, 16)], dtype=np.float64)
    contacts = [float((pair_dist < r).sum()) for r in (5.0, 8.0, 10.0, 12.0)]

    # Backbone-like proxy: count of pairs in the hydrogen-bond distance window.
    hbond = float(((pair_dist > 2.5) & (pair_dist < 3.6)).sum())

    feats = np.concatenate([
        np.array([rg, max_dist, anisotropy, n], dtype=np.float64),
        eigvals / (rg ** 2 + 1e-8),
        counts / n,
        np.asarray(contacts, dtype=np.float64) / max(1.0, n),
        np.array([elec / 1e4, lj / 1e3, hbond / max(1.0, n)], dtype=np.float64),
        np.array([len(elements), pair_dist.mean() if pair_dist.size else 0.0], dtype=np.float64),
    ])
    out = np.zeros(24, dtype=np.float32)
    out[: min(24, feats.shape[0])] = feats[:24].astype(np.float32)
    return out


DESCRIPTOR_DIM = 24


# --------------------------------------------------------------------------- #
# Torch Dataset
# --------------------------------------------------------------------------- #


@dataclass
class Sample:
    """One (structure, energy) training example."""

    protein: str
    frame: int
    source: str
    y: float
    n_atoms: int


class ConformerDataset(Dataset):
    """PyTorch dataset over (protein, frame) pairs.

    Featurisation happens in ``__getitem__`` rather than up front: the graphs are
    cheap to build, and keeping only compact coordinate arrays in memory keeps the
    whole dataset at a few tens of megabytes.
    """

    def __init__(
        self,
        ensembles: dict,
        index: Sequence[tuple[str, int]],
        y_values: np.ndarray,
        cfg: FeatureConfig,
        *,
        descriptor_mode: bool = False,
        cache: dict | None = None,
    ):
        self.ensembles = ensembles
        self.index = list(index)
        self.y = np.asarray(y_values, dtype=np.float32)
        self.cfg = cfg
        self.descriptor_mode = descriptor_mode
        #: Optional precomputed {(protein, frame): GraphArrays} from graphcache.
        #: Featurisation is the single most expensive part of an epoch, so the
        #: same graphs are built once and reused for every epoch.
        self.cache = cache
        if len(self.index) != len(self.y):
            raise ValueError("index and target arrays must have equal length")

    def __len__(self) -> int:
        return len(self.index)

    def sample_meta(self, i: int) -> Sample:
        protein, frame = self.index[i]
        ens = self.ensembles[protein]
        return Sample(
            protein=protein,
            frame=int(frame),
            source=str(ens.source[frame]) if ens.source.size else "",
            y=float(self.y[i]),
            n_atoms=int(ens.n_atoms),
        )

    def __getitem__(self, i: int) -> dict:
        protein, frame = self.index[i]
        ens = self.ensembles[protein]
        coords = np.asarray(ens.coords[frame], dtype=np.float64)
        elements = [str(e) for e in ens.elements]
        if self.descriptor_mode:
            return {
                "x": torch.from_numpy(global_descriptors(coords, elements)),
                "y": torch.tensor(float(self.y[i])),
                "protein": protein,
                "frame": int(frame),
                "source": str(ens.source[frame]) if ens.source.size else "",
            }
        graph = None
        if self.cache is not None:
            graph = self.cache.get((protein, int(frame)))
        if graph is None:
            graph = frame_to_graph(coords, elements, self.cfg)
        return {
            "z": torch.from_numpy(graph.z),
            "pos": torch.from_numpy(graph.pos),
            "edge_index": torch.from_numpy(graph.edge_index),
            "edge_attr": torch.from_numpy(graph.edge_attr),
            "y": torch.tensor(float(self.y[i])),
            "protein": protein,
            "frame": int(frame),
            "source": str(ens.source[frame]) if ens.source.size else "",
        }


def collate_graphs(batch: Sequence[dict]) -> dict:
    """Concatenate several small graphs into one disconnected batch graph.

    Batching graphs this way (rather than padding to a rectangle) means the
    model only ever processes real atom pairs, and the ``batch`` vector tells
    the readout which atoms belong to which molecule.
    """
    z_list, pos_list, ei_list, ea_list, batch_idx = [], [], [], [], []
    ys, proteins, frames, sources = [], [], [], []
    offset = 0
    for b, item in enumerate(batch):
        n = item["z"].shape[0]
        z_list.append(item["z"])
        pos_list.append(item["pos"])
        ei_list.append(item["edge_index"] + offset)
        ea_list.append(item["edge_attr"])
        batch_idx.append(torch.full((n,), b, dtype=torch.long))
        ys.append(item["y"])
        proteins.append(item["protein"])
        frames.append(item["frame"])
        sources.append(item["source"])
        offset += n
    return {
        "z": torch.cat(z_list),
        "pos": torch.cat(pos_list),
        "edge_index": torch.cat(ei_list, dim=1),
        "edge_attr": torch.cat(ea_list, dim=0),
        "batch": torch.cat(batch_idx),
        "y": torch.stack(ys),
        "n_graphs": len(batch),
        "protein": proteins,
        "frame": frames,
        "source": sources,
    }


def collate_descriptors(batch: Sequence[dict]) -> dict:
    """Collate the fixed-length descriptor baseline."""
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "n_graphs": len(batch),
        "protein": [b["protein"] for b in batch],
        "frame": [b["frame"] for b in batch],
        "source": [b["source"] for b in batch],
    }
