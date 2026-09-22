"""Assemble labelled samples and split them *without leaking information*.

The most important idea in this module
--------------------------------------
Splitting a dataset of protein conformations **by frame** is a trap.  Frames
from the same protein are near-duplicates: two MD snapshots 0.4 ps apart differ
by fractions of an Angstrom.  A frame-level random split therefore puts
near-copies of every test structure into the training set, the test error looks
excellent, and the model has learned nothing that transfers to a new protein.

We split **by protein** instead: three proteins are held out entirely and are
never seen during training or model selection.  The gap between the two splits
is reported explicitly by :mod:`pdbenergy.evaluate`, because it measures exactly
how much the leaky split was flattering the model.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .config import Config, FeatureConfig
from .ensemble import ProteinEnsemble
from .features import ConformerDataset

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_ensembles(interim_dir: str, *, verbose: bool = True) -> dict[str, ProteinEnsemble]:
    """Load every ``<PDBID>.npz`` ensemble written by ``ensemble.build_all``."""
    if not os.path.isdir(interim_dir):
        raise FileNotFoundError(
            f"no ensemble directory at {interim_dir!r}; run the `ensemble` step first"
        )
    out: dict[str, ProteinEnsemble] = {}
    for name in sorted(os.listdir(interim_dir)):
        if not name.endswith(".npz"):
            continue
        pdb_id = name[:-4]
        ensemble = ProteinEnsemble.load(os.path.join(interim_dir, name))
        # ProteinEnsemble.load derives the id from the file name; keep it explicit.
        ensemble.pdb_id = pdb_id
        if len(ensemble) == 0:
            continue
        out[pdb_id] = ensemble
        if verbose:
            print(
                f"  loaded {pdb_id}: {len(ensemble):>4} frames, "
                f"{ensemble.n_atoms:>4} atoms, "
                f"dE 0..{ensemble.relative_energy.max():.1f} kcal/mol",
                flush=True,
            )
    if not out:
        raise FileNotFoundError(f"no .npz ensembles found in {interim_dir!r}")
    return out


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def split_proteins(
    proteins: Sequence[str],
    *,
    seed: int = 0,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> dict[str, list[str]]:
    """Deterministically partition protein IDs into train/val/test.

    Deterministic because the shuffle is driven by an explicitly seeded
    ``numpy.random.default_rng``: the same seed gives the same split forever,
    which is what makes a reported number reproducible.
    """
    proteins = sorted(proteins)
    if len(proteins) < 3:
        # Too few proteins to hold any out; degrade gracefully and say so.
        return {"train": list(proteins), "val": list(proteins), "test": list(proteins)}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(proteins))
    n_test = max(1, int(round(len(proteins) * test_fraction)))
    n_val = max(1, int(round(len(proteins) * val_fraction)))
    if n_test + n_val >= len(proteins):
        n_test = 1
        n_val = 1
    test = [proteins[i] for i in order[:n_test]]
    val = [proteins[i] for i in order[n_test:n_test + n_val]]
    train = [proteins[i] for i in order[n_test + n_val:]]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #


@dataclass
class DataBundle:
    """Everything the training loop needs, plus the provenance of every split."""

    ensembles: dict[str, ProteinEnsemble]
    cfg: Config
    rows: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    y_raw: dict[str, np.ndarray] = field(default_factory=dict)
    split_mode: str = "protein"
    protein_splits: dict[str, list[str]] = field(default_factory=dict)
    norm_mean: float = 0.0
    norm_std: float = 1.0
    #: For the leaky "frame" mode we keep the *honest* protein split so the
    #: evaluation can also report the leak-free number for the same model.
    leak_free_proteins: dict[str, list[str]] = field(default_factory=dict)

    # -- target handling ---------------------------------------------------- #
    @property
    def target(self) -> str:
        return self.cfg.train.target

    def raw_target(self, protein: str, frame: int) -> float:
        ens = self.ensembles[protein]
        if self.target == "absolute":
            return float(ens.energy_total[frame])
        return float(ens.energy_total[frame] - ens.reference_energy)

    def normalise(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=np.float32) - self.norm_mean) / self.norm_std

    def denormalise(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float64) * self.norm_std + self.norm_mean

    # -- torch datasets ----------------------------------------------------- #
    def dataset(
        self, split: str, *, descriptor_mode: bool = False, cache: dict | None = None
    ) -> ConformerDataset:
        rows = self.rows[split]
        y_normalised = self.normalise(self.y_raw[split])
        return ConformerDataset(
            self.ensembles, rows, y_normalised, self.cfg.features,
            descriptor_mode=descriptor_mode, cache=cache,
        )

    def all_rows(self) -> list[tuple[str, int]]:
        """Every (protein, frame) pair across all splits, de-duplicated."""
        seen: dict[tuple[str, int], None] = {}
        for split in SPLIT_NAMES:
            for row in self.rows.get(split, []):
                seen[(row[0], int(row[1]))] = None
        return list(seen)

    def feature_config(self) -> FeatureConfig:
        return self.cfg.features

    def summary(self) -> dict:
        out = {
            "target": self.target,
            "split_mode": self.split_mode,
            "n_proteins": len(self.ensembles),
            "n_samples": int(sum(len(v) for v in self.rows.values())),
            "norm_mean": float(self.norm_mean),
            "norm_std": float(self.norm_std),
            "proteins": {k: self.protein_splits.get(k, []) for k in SPLIT_NAMES},
            "samples": {k: len(v) for k, v in self.rows.items()},
            "per_protein": {
                p: {
                    "n_frames": len(ens),
                    "n_atoms": ens.n_atoms,
                    # Count residues from the prepared topology's residue index
                    # rather than from the stored one-letter sequence: the
                    # sequence field is built from the raw file, which also
                    # contains waters and ions, so it over-counts.
                    "sequence_length": (
                        int(len(np.unique(ens.residue_index)))
                        if ens.residue_index.size else len(ens.sequence)
                    ),
                    "reference_energy_kcal": float(ens.reference_energy),
                    "dE_max_kcal": float(ens.relative_energy.max()),
                    "sources": ens.meta.get("source_counts", {}),
                }
                for p, ens in sorted(self.ensembles.items())
            },
        }
        return out

    # -- persistence -------------------------------------------------------- #
    def save_rows(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "rows": {k: [[p, int(f)] for p, f in v] for k, v in self.rows.items()},
            "y_raw": {k: np.asarray(v).tolist() for k, v in self.y_raw.items()},
            "norm_mean": self.norm_mean,
            "norm_std": self.norm_std,
            "protein_splits": self.protein_splits,
            "split_mode": self.split_mode,
            "target": self.target,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return path


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def _all_rows(ensembles: dict[str, ProteinEnsemble]) -> dict[str, list[tuple[str, int]]]:
    return {
        p: [(p, f) for f in range(len(ens))]
        for p, ens in sorted(ensembles.items())
    }


def build_bundle(
    ensembles: dict[str, ProteinEnsemble],
    cfg: Config,
    *,
    split_mode: str | None = None,
    verbose: bool = True,
) -> DataBundle:
    """Create train/val/test splits, targets and normalisation statistics."""
    mode = split_mode or getattr(cfg.train, "split_mode", "protein")
    proteins = sorted(ensembles)
    protein_splits = split_proteins(
        proteins,
        seed=cfg.train.seed,
        val_fraction=cfg.train.val_fraction,
        test_fraction=cfg.train.test_fraction,
    )

    bundle = DataBundle(
        ensembles=ensembles, cfg=cfg, split_mode=mode, protein_splits=protein_splits,
        leak_free_proteins=protein_splits,
    )

    if mode == "protein":
        rows: dict[str, list[tuple[str, int]]] = {k: [] for k in SPLIT_NAMES}
        for split, ids in protein_splits.items():
            for p in ids:
                rows[split].extend((p, f) for f in range(len(ensembles[p])))
    elif mode == "frame":
        # Deliberately leaky: individual frames are shuffled across the split.
        all_rows = [(p, f) for p in proteins for f in range(len(ensembles[p]))]
        rng = np.random.default_rng(cfg.train.seed)
        idx = rng.permutation(len(all_rows))
        n_test = max(1, int(round(len(all_rows) * cfg.train.test_fraction)))
        n_val = max(1, int(round(len(all_rows) * cfg.train.val_fraction)))
        test_idx, val_idx, train_idx = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
        rows = {
            "train": [all_rows[i] for i in train_idx],
            "val": [all_rows[i] for i in val_idx],
            "test": [all_rows[i] for i in test_idx],
        }
    else:
        raise ValueError(f"unknown split_mode {mode!r}; use 'protein' or 'frame'")

    bundle.rows = rows
    bundle.y_raw = {
        split: np.asarray([bundle.raw_target(p, f) for p, f in rows[split]], dtype=np.float64)
        for split in SPLIT_NAMES
    }

    train_y = bundle.y_raw["train"]
    bundle.norm_mean = float(np.mean(train_y)) if train_y.size else 0.0
    bundle.norm_std = float(np.std(train_y)) if train_y.size else 1.0
    if not np.isfinite(bundle.norm_std) or bundle.norm_std < 1e-8:
        bundle.norm_std = 1.0

    if verbose:
        print(
            f"  split_mode={mode} target={cfg.train.target}\n"
            f"  train proteins ({len(protein_splits['train'])}): {protein_splits['train']}\n"
            f"  val   proteins ({len(protein_splits['val'])}): {protein_splits['val']}\n"
            f"  test  proteins ({len(protein_splits['test'])}): {protein_splits['test']}\n"
            f"  samples: train={len(rows['train'])} val={len(rows['val'])} test={len(rows['test'])}\n"
            f"  target normalisation: mean={bundle.norm_mean:.3f} std={bundle.norm_std:.3f}",
            flush=True,
        )
    return bundle


def collate_for(descriptor_mode: bool):
    """Pick the right collate_fn for the model family."""
    from .features import collate_descriptors, collate_graphs

    return collate_descriptors if descriptor_mode else collate_graphs
