"""Generate conformational ensembles and label them with physics energies.

A machine-learned potential is only as good as the region of configuration
space its training set covers.  This module builds that coverage deliberately
from four complementary sources:

===============  ==================================================================
source           what it explores
===============  ==================================================================
``nmr``          experimentally observed models of the *same* protein (deposited
                 NMR ensembles contain 10-40 real conformers - free diversity)
``torsion``      random rotations about rotatable single bonds.  Bond lengths and
                 angles are preserved *exactly*, so the energy change is purely
                 conformational: torsions, van der Waals, electrostatics.
``torsion_min``  the same, energy-minimised -> local minima, low energy region
``md``           Langevin molecular dynamics at 300 K / 450 K.  Thermally
                 accessible states, i.e. the configurations that actually occur.
===============  ==================================================================

Why not just add Gaussian noise to Cartesian coordinates?
--------------------------------------------------------
Because a C-H bond is only 1.09 A long.  Independent Gaussian displacement with
sigma = 1 A stretches every bond far beyond its harmonic range, so ~99.9% of the
resulting energy is bond-stretching energy.  The model would then spend its
capacity fitting a *bond-length* function it cannot even observe properly, and
the chemically interesting energy differences would be buried.  Rotating about
bonds is the physically meaningful way to move a molecule.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import EnsembleConfig, LabelConfig
from .labels import ENERGY_TERMS, OpenMMEnergyEngine, PreparedSystem
from .pdbio import Structure, protein_only, read_pdb
from .prepare import write_json


# --------------------------------------------------------------------------- #
# Molecular graph utilities
# --------------------------------------------------------------------------- #


def bond_graph(topology) -> list[tuple[int, int]]:
    """Return the bonded atom-index pairs of an OpenMM topology."""
    return [(b[0].index, b[1].index) for b in topology.bonds()]


def adjacency(n_atoms: int, bonds: Sequence[tuple[int, int]]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _in_ring(adj: list[list[int]], a: int, b: int, max_depth: int = 6) -> bool:
    """True if the a-b bond lies on a ring, i.e. a second short a->b path exists.

    A depth-limited breadth-first search with the a-b edge removed answers this
    cheaply and correctly for the small rings found in proteins (3-7 membered).
    """
    seen = {a}
    frontier = [a]
    for _ in range(max_depth):
        nxt: list[int] = []
        for u in frontier:
            for v in adj[u]:
                if (u == a and v == b) or (u == b and v == a):
                    continue
                if v == b:
                    return True
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        if not nxt:
            return False
        frontier = nxt
    return False


def find_rotatable_bonds(topology, max_bonds: int | None = None) -> list[tuple[int, int]]:
    """Enumerate rotatable single bonds between two heavy atoms.

    A bond is kept when it is

    * between two non-hydrogen atoms,
    * not part of a ring (ring bonds cannot rotate),
    * not terminal (rotating a terminal group changes nothing),
    * not the peptide C-N bond (partial double bond character makes the peptide
      plane rigid - this is the single most important rigidity in proteins).

    Returns pairs ``(a, b)`` where ``a`` stays put and ``b``'s side rotates.
    """
    atoms = list(topology.atoms())
    n = len(atoms)
    bonds = bond_graph(topology)
    adj = adjacency(n, bonds)

    is_h = [a.element is not None and a.element.symbol.upper() in ("H", "D") for a in atoms]
    heavy_degree = [sum(0 if is_h[v] else 1 for v in adj[i]) for i in range(n)]

    def heavy_component_size(root: int, blocked_a: int, blocked_b: int) -> int:
        """Number of heavy atoms reachable from ``root`` without the cut edge."""
        seen = {root}
        stack = [root]
        count = 0
        while stack:
            u = stack.pop()
            if not is_h[u]:
                count += 1
            for v in adj[u]:
                if (u == blocked_a and v == blocked_b) or (u == blocked_b and v == blocked_a):
                    continue
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return count

    out: list[tuple[int, int]] = []
    for i, j in bonds:
        if is_h[i] or is_h[j]:
            continue
        if heavy_degree[i] < 2 or heavy_degree[j] < 2:
            continue
        if _in_ring(adj, i, j):
            continue
        # Peptide bond: backbone C of residue k to backbone N of residue k+1.
        ai, aj = atoms[i], atoms[j]
        names = {ai.name.strip(), aj.name.strip()}
        if names == {"C", "N"} and ai.residue.index != aj.residue.index:
            continue
        # Both sides must retain at least two heavy atoms, else nothing rotates.
        if heavy_component_size(j, i, j) < 2 or heavy_component_size(i, i, j) < 2:
            continue
        out.append((i, j))

    if max_bonds is not None and len(out) > max_bonds:
        # Prefer backbone torsions (they dominate conformational energy) but keep
        # a deterministic, reproducible subset.
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(out), size=max_bonds, replace=False))
        out = [out[k] for k in idx]
    return out


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues' rotation formula for a rotation of ``angle_rad`` about ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def rotate_torsion(
    positions: np.ndarray,
    adj: list[list[int]],
    a: int,
    b: int,
    angle_rad: float,
) -> np.ndarray:
    """Rotate the ``b``-side subtree about the a-b axis by ``angle_rad``.

    Because the rotation axis passes through the bond, every bond length and
    bond angle is preserved to machine precision: only the dihedral changes.
    """
    moved = _subtree_atoms(adj, a, b)
    origin = positions[a]
    axis = positions[b] - positions[a]
    R = rotation_matrix(axis, angle_rad)
    out = positions.copy()
    out[moved] = origin + (positions[moved] - origin) @ R.T
    return out


def _subtree_atoms(adj: list[list[int]], a: int, b: int) -> np.ndarray:
    """All atoms on ``b``'s side of the a-b bond (inclusive of ``b``)."""
    seen = {b}
    stack = [b]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if (u == a and v == b) or (u == b and v == a):
                continue
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return np.fromiter(sorted(seen), dtype=np.int64)


# --------------------------------------------------------------------------- #
# Ensemble container
# --------------------------------------------------------------------------- #


@dataclass
class ProteinEnsemble:
    """All labelled conformations collected for one protein.

    ``coords`` holds **all atoms** (hydrogens included) because the energy being
    learned is an all-atom force-field energy: the same heavy atoms with
    different hydrogens are genuinely different energies.
    """

    pdb_id: str
    coords: np.ndarray            # (S, N, 3) float32, Angstrom
    elements: np.ndarray          # (N,) unicode
    atom_names: np.ndarray        # (N,) unicode
    residue_index: np.ndarray     # (N,) int32
    residue_names: np.ndarray     # (N,) unicode
    energy_total: np.ndarray      # (S,) float64 kcal/mol
    energy_terms: dict[str, np.ndarray] = field(default_factory=dict)
    source: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="<U12"))
    reference_energy: float = float("nan")
    sequence: str = ""
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.coords.shape[1])

    @property
    def relative_energy(self) -> np.ndarray:
        """E - min(E): the conformational energy on a per-protein zero."""
        return self.energy_total - float(np.min(self.energy_total))

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str | os.PathLike) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
        arrays = {
            "coords": self.coords.astype(np.float32),
            "elements": self.elements.astype("U2"),
            "atom_names": self.atom_names.astype("U4"),
            "residue_index": self.residue_index.astype(np.int32),
            "residue_names": self.residue_names.astype("U3"),
            "energy_total": self.energy_total.astype(np.float64),
            "source": self.source.astype("U16"),
            "reference_energy": np.float64(self.reference_energy),
            "sequence": np.asarray(self.sequence),
            "meta_json": np.asarray(json.dumps(self.meta, default=str)),
        }
        for term, values in self.energy_terms.items():
            arrays[f"term_{term}"] = np.asarray(values, dtype=np.float64)
        np.savez_compressed(str(path), **arrays)
        return str(path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ProteinEnsemble":
        with np.load(str(path), allow_pickle=False) as data:
            terms = {
                k[len("term_"):]: data[k] for k in data.files if k.startswith("term_")
            }
            meta_raw = str(data["meta_json"]) if "meta_json" in data.files else "{}"
            return cls(
                pdb_id=os.path.basename(str(path)).split(".")[0],
                coords=data["coords"],
                elements=data["elements"],
                atom_names=data["atom_names"],
                residue_index=data["residue_index"],
                residue_names=data["residue_names"],
                energy_total=data["energy_total"],
                energy_terms=terms,
                source=data["source"] if "source" in data.files else np.array([], dtype="U16"),
                reference_energy=float(data["reference_energy"]),
                sequence=str(data["sequence"]) if "sequence" in data.files else "",
                meta=json.loads(meta_raw) if meta_raw else {},
            )


# --------------------------------------------------------------------------- #
# Ensemble construction
# --------------------------------------------------------------------------- #


def _frame(positions_all_atom: np.ndarray, prepared: PreparedSystem) -> np.ndarray:
    return np.asarray(positions_all_atom, dtype=np.float64)


def build_ensemble(
    structure: Structure,
    pdb_id: str,
    *,
    engine: OpenMMEnergyEngine,
    ensemble_cfg: EnsembleConfig,
    all_models: Sequence[Structure] | None = None,
    workdir: str | None = None,
    verbose: bool = True,
) -> ProteinEnsemble:
    """Prepare one protein and label a whole conformational ensemble.

    Returns a :class:`ProteinEnsemble` whose ``energy_total`` are absolute
    AMBER14+GBn2 energies in kcal/mol and whose ``reference_energy`` is the
    lowest energy found (the per-protein zero of the relative scale).
    """
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    rng = np.random.default_rng(ensemble_cfg.seed)

    prepared = engine.prepare(structure, pdb_id, workdir=workdir)
    n_atoms = prepared.n_atoms_prepared
    bonds = bond_graph(prepared.topology)
    adj = adjacency(n_atoms, bonds)
    rotatable = find_rotatable_bonds(prepared.topology)
    log(f"  [{pdb_id}] prepared: {n_atoms} atoms ({prepared.n_atoms_heavy} heavy), "
        f"{len(bonds)} bonds, {len(rotatable)} rotatable bonds")

    context, _ = engine.make_context(prepared)

    # ---- 1. reference: energy-minimised prepared structure ---------------- #
    engine.set_positions(context, prepared.positions_nm * 10.0)
    e_native = engine.minimise(context, ensemble_cfg.reference_minimise_iterations)
    native_pos = (
        context.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(engine.openmm[2].angstrom)
    )
    native_pos = np.array(native_pos, dtype=np.float64)

    # ---- screening collector --------------------------------------------- #
    # Every candidate is scored immediately and rejected when it exceeds the
    # reference by `max_relative_energy`.  This is what keeps 1/r^12 clashes out
    # of the dataset instead of letting them dominate the loss.
    limit = float(ensemble_cfg.max_relative_energy)
    md_limit = float(ensemble_cfg.md_max_relative_energy)
    frames: list[np.ndarray] = []
    sources: list[str] = []
    energies: list[float] = []
    rejected = 0
    best_energy = float(e_native)

    def energy_of(pos: np.ndarray) -> float:
        engine.set_positions(context, pos)
        return engine.energy_kcal(context)

    def consider(pos: np.ndarray, source: str, cap: float | None = None) -> bool:
        nonlocal rejected, best_energy
        threshold = limit if cap is None else cap
        value = energy_of(pos)
        if not np.isfinite(value) or value - e_native > threshold:
            rejected += 1
            return False
        frames.append(pos)
        sources.append(source)
        energies.append(value)
        best_energy = min(best_energy, value)
        return True

    log(f"  [{pdb_id}] reference (minimised) energy = {e_native:.2f} kcal/mol "
        f"| screening limit +{limit:.0f} kcal/mol")

    # ---- 2. NMR models: real experimental conformers ---------------------- #
    if ensemble_cfg.use_nmr_models and all_models and ensemble_cfg.max_nmr_models > 0:
        nmr_added = 0
        nmr_attempted = 0
        for model in all_models[1:]:
            # Bound the *attempts*, not the successes: if a relaxed model still
            # lands above the screening limit we would otherwise keep walking
            # through all 30-40 deposited models, paying for a minimisation each
            # time.  Cost must not depend on how well the screening happens to go.
            if nmr_attempted >= ensemble_cfg.max_nmr_models:
                break
            heavy = protein_only(model, keep_hydrogens=False)
            if len(heavy) != prepared.n_atoms_heavy:
                # Some models in an NMR ensemble have missing atoms; using them
                # would mean scoring invented coordinates.
                continue
            nmr_attempted += 1
            try:
                topology, positions_nm = engine.protonate(
                    heavy, f"{pdb_id}_nmr{nmr_attempted}", workdir=workdir
                )
            except Exception as exc:
                log(f"  [{pdb_id}] NMR model skipped: {type(exc).__name__}: {exc}")
                continue
            if not _same_atom_layout(topology, prepared):
                log(f"  [{pdb_id}] NMR model skipped: atom layout differs")
                continue
            engine.set_positions(context, positions_nm * 10.0)
            engine.minimise(context, ensemble_cfg.nmr_minimise_iterations)
            pos = np.array(
                context.getState(getPositions=True)
                .getPositions(asNumpy=True)
                .value_in_unit(engine.openmm[2].angstrom),
                dtype=np.float64,
            )
            if consider(pos, "nmr"):
                nmr_added += 1
        if nmr_added:
            log(f"  [{pdb_id}] incorporated {nmr_added} NMR models "
                f"(from {nmr_attempted} attempted)")

    # ---- 3. torsional sampling ------------------------------------------- #
    if rotatable:
        n_levels = max(1, len(ensemble_cfg.torsion_levels))
        target = max(1, ensemble_cfg.n_torsion_per_level)
        for level in range(n_levels):
            frac = level / max(1, n_levels - 1) if n_levels > 1 else 1.0
            span = np.deg2rad(
                ensemble_cfg.torsion_min_deg
                + frac * (ensemble_cfg.torsion_max_deg - ensemble_cfg.torsion_min_deg)
            )
            accepted, attempts = 0, 0
            max_attempts = target * max(1, ensemble_cfg.max_attempt_factor)
            while accepted < target and attempts < max_attempts:
                attempts += 1
                pos = native_pos.copy()
                n_moves = int(rng.integers(ensemble_cfg.moves_min, ensemble_cfg.moves_max + 1))
                for _ in range(max(1, n_moves)):
                    a, b = rotatable[int(rng.integers(0, len(rotatable)))]
                    angle = float(rng.uniform(-span, span))
                    pos = rotate_torsion(pos, adj, a, b, angle)
                if ensemble_cfg.cartesian_jitter > 0:
                    pos = pos + rng.normal(0.0, ensemble_cfg.cartesian_jitter, size=pos.shape)
                if rng.random() < ensemble_cfg.minimise_fraction:
                    engine.set_positions(context, pos)
                    engine.minimise(context, ensemble_cfg.partial_minimise_iterations)
                    pos = np.array(
                        context.getState(getPositions=True)
                        .getPositions(asNumpy=True)
                        .value_in_unit(engine.openmm[2].angstrom),
                        dtype=np.float64,
                    )
                    accepted += int(consider(pos, "torsion_min"))
                else:
                    accepted += int(consider(pos, "torsion"))
            keep_rate = accepted / max(1, attempts)
            log(
                f"  [{pdb_id}] torsion level {level + 1}/{n_levels} "
                f"(+-{np.rad2deg(span):.0f} deg): kept {accepted}/{attempts} "
                f"({keep_rate:.0%})"
            )
    else:
        log(f"  [{pdb_id}] warning: no rotatable bonds found, torsion sampling skipped")

    # ---- 4. molecular dynamics ------------------------------------------- #
    md_ps = ensemble_cfg.md_ps_per_temperature
    if md_ps > 0 and ensemble_cfg.md_temperatures:
        dt_fs = ensemble_cfg.md_timestep_fs
        n_steps = int(round(md_ps * 1000.0 / dt_fs))
        save_every = max(1, int(round(ensemble_cfg.md_save_every_ps * 1000.0 / dt_fs)))
        for temperature in ensemble_cfg.md_temperatures:
            md_frames = engine.run_md(
                prepared,
                temperature_k=float(temperature),
                n_steps=n_steps,
                save_every=save_every,
                seed=int(rng.integers(0, 2**31 - 1)),
                start_positions_angstrom=native_pos,
            )
            kept = sum(
                int(consider(pos, f"md{int(temperature)}K", md_limit)) for pos in md_frames
            )
            log(f"  [{pdb_id}] MD {temperature:.0f} K: kept {kept}/{len(md_frames)} frames")

    # ---- 5. energy decomposition for the kept frames --------------------- #
    energy_array = np.asarray(energies, dtype=np.float64)
    terms = {term: np.zeros(len(frames), dtype=np.float64) for term in ENERGY_TERMS}
    for k, pos in enumerate(frames):
        engine.set_positions(context, pos)
        for term, value in engine.energy_terms_kcal(context).items():
            terms[term][k] = value
    del context

    if not frames:
        raise RuntimeError(
            f"{pdb_id}: every generated frame was rejected; raise "
            f"max_relative_energy (currently {limit}) or reduce torsion angles"
        )
    if rejected:
        log(f"  [{pdb_id}] rejected {rejected} clashing frames above +{limit:.0f} kcal/mol")

    coords = np.stack(frames).astype(np.float32)
    reference = float(best_energy)

    # Sequence for display/reporting.  Use the cleaned protein view: the raw file
    # also contains waters and ions, and `Structure.sequence()` would count each
    # of those as a residue (they show up as 'X'), inflating the length by ~50%.
    seq = "".join(protein_only(structure).sequence().values())
    ensemble = ProteinEnsemble(
        pdb_id=pdb_id,
        coords=coords,
        elements=np.asarray(_all_atom_elements(prepared), dtype="U2"),
        atom_names=np.asarray(_all_atom_names(prepared), dtype="U4"),
        residue_index=_all_atom_residue_index(prepared),
        residue_names=np.asarray(_all_atom_residue_names(prepared), dtype="U3"),
        energy_total=energy_array,
        energy_terms=terms,
        source=np.asarray(sources, dtype="U16"),
        reference_energy=reference,
        sequence=seq,
        meta={
            "n_atoms_prepared": n_atoms,
            "n_atoms_heavy": prepared.n_atoms_heavy,
            "n_bonds": len(bonds),
            "n_rotatable": len(rotatable),
            "native_energy_kcal": float(e_native),
            "n_rejected": int(rejected),
            "source_counts": {s: sources.count(s) for s in sorted(set(sources))},
        },
    )
    return ensemble


def _all_atom_elements(prepared: PreparedSystem) -> list[str]:
    atoms = list(prepared.topology.atoms())
    return [(a.element.symbol.upper() if a.element is not None else "X") for a in atoms]


def _all_atom_names(prepared: PreparedSystem) -> list[str]:
    return [a.name.strip() for a in prepared.topology.atoms()]


def _all_atom_residue_index(prepared: PreparedSystem) -> np.ndarray:
    return np.asarray([a.residue.index for a in prepared.topology.atoms()], dtype=np.int32)


def _all_atom_residue_names(prepared: PreparedSystem) -> list[str]:
    return [a.residue.name for a in prepared.topology.atoms()]


def _same_atom_layout(topology, prepared: PreparedSystem) -> bool:
    """True when two topologies list exactly the same atoms, in the same order.

    Repreparing an NMR model re-derives its hydrogens from scratch, which is the
    only way to get chemically sensible X-H bond lengths for that model.  But the
    result is only usable as a frame of *this* protein's ensemble if PDBFixer
    produced the identical atom ordering; otherwise the energies would refer to a
    different particle list. Comparing the element sequence is a cheap, decisive
    check.
    """
    got = [
        (a.element.symbol.upper() if a.element is not None else "X")
        for a in topology.atoms()
    ]
    expected = [
        (a.element.symbol.upper() if a.element is not None else "X")
        for a in prepared.topology.atoms()
    ]
    return got == expected


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #


def build_one(
    pdb_id: str,
    path: str,
    out_dir: str,
    *,
    engine: OpenMMEnergyEngine,
    ensemble_cfg: EnsembleConfig,
    overwrite: bool = False,
    verbose: bool = True,
) -> tuple[str, str | None, dict | None]:
    """Build and save one protein's ensemble.  Returns (id, npz_path, summary)."""
    target = os.path.join(out_dir, f"{pdb_id}.npz")
    if os.path.exists(target) and not overwrite:
        if verbose:
            print(f"[{pdb_id}] cached -> {target}", flush=True)
        return pdb_id, target, None
    models = read_pdb(path)
    if not models:
        print(f"[{pdb_id}] SKIP: no atoms", flush=True)
        return pdb_id, None, None
    try:
        ens = build_ensemble(
            models[0], pdb_id,
            engine=engine, ensemble_cfg=ensemble_cfg,
            all_models=models, workdir=out_dir,
            verbose=verbose,
        )
    except Exception as exc:  # one bad entry must not kill the whole run
        print(f"[{pdb_id}] FAILED: {type(exc).__name__}: {exc}", flush=True)
        return pdb_id, None, None
    ens.save(target)
    summary = {
        "pdb_id": pdb_id,
        "n_frames": len(ens),
        "n_atoms": ens.n_atoms,
        "reference_energy_kcal": ens.reference_energy,
        "energy_min_kcal": float(np.min(ens.energy_total)),
        "energy_max_kcal": float(np.max(ens.energy_total)),
        "sources": {s: int((ens.source == s).sum()) for s in sorted(set(ens.source))},
        "sequence": ens.sequence,
    }
    write_json(os.path.join(out_dir, f"{pdb_id}.json"), summary)
    if verbose:
        print(
            f"[{pdb_id}] {len(ens)} frames, dE range "
            f"{ens.relative_energy.min():.2f}..{ens.relative_energy.max():.2f} kcal/mol",
            flush=True,
        )
    return pdb_id, target, summary


# -- parallel driver (one process per protein) ------------------------------ #

_WORKER_ENGINE: OpenMMEnergyEngine | None = None
_WORKER_CFG: tuple | None = None


def _worker_init(label_cfg: LabelConfig, prepare_cfg, ensemble_cfg: EnsembleConfig, threads: int) -> None:
    """Runs once per pool worker: build the engine there, not in the parent."""
    global _WORKER_ENGINE, _WORKER_CFG
    _WORKER_ENGINE = OpenMMEnergyEngine(label_cfg, prepare_cfg, threads=threads)
    _WORKER_CFG = (label_cfg, prepare_cfg, ensemble_cfg, threads)


def _worker_run(job: tuple[str, str, str, bool]) -> tuple[str, str | None, dict | None]:
    pdb_id, path, out_dir, overwrite = job
    label_cfg, prepare_cfg, ensemble_cfg, _threads = _WORKER_CFG
    return build_one(
        pdb_id, path, out_dir,
        engine=_WORKER_ENGINE, ensemble_cfg=ensemble_cfg,
        overwrite=overwrite, verbose=True,
    )


def build_all(
    pdb_paths: dict[str, str],
    *,
    out_dir: str,
    ensemble_cfg: EnsembleConfig,
    label_cfg: LabelConfig,
    prepare_cfg=None,
    threads: int = 1,
    workers: int = 1,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict[str, str]:
    """Build ensembles for many proteins.

    Each protein is written to ``<out_dir>/<PDBID>.npz``, so an interrupted run
    resumes where it stopped (pass ``overwrite=False``, the default).

    ``workers > 1`` runs several proteins concurrently, each with ``threads``
    OpenMM threads.  Total CPU demand is ``workers * threads``, so keep that at
    or below the number of physical cores.
    """
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(pdb_id, path, out_dir, overwrite) for pdb_id, path in sorted(pdb_paths.items())]

    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        written: dict[str, str] = {}
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(label_cfg, prepare_cfg, ensemble_cfg, threads),
        ) as pool:
            for pdb_id, target, _summary in pool.map(_worker_run, jobs):
                if target:
                    written[pdb_id] = target
        return written

    engine = OpenMMEnergyEngine(label_cfg, prepare_cfg, threads=threads, verbose=verbose)
    written = {}
    for pdb_id, path, _out, ow in jobs:
        pid, target, _summary = build_one(
            pdb_id, path, out_dir, engine=engine, ensemble_cfg=ensemble_cfg,
            overwrite=ow, verbose=verbose,
        )
        if target:
            written[pid] = target
    return written
