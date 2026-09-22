"""Configuration objects for the whole pipeline.

Everything the pipeline does is controlled by one JSON/YAML-serialisable
dictionary so that a run is fully reproducible: the config that produced a
checkpoint is written next to it (``outputs/<run>/config.json``).
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class PathConfig:
    """Where things live."""

    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    outputs_dir: str = "outputs"


@dataclass
class PrepareConfig:
    """How raw PDB entries are turned into force-field-ready structures."""

    #: Discard waters/ions/ligands and keep only standard amino acids.
    protein_only: bool = True
    #: Let PDBFixer add missing heavy atoms and hydrogens (pH-dependent).
    add_hydrogens: bool = True
    ph: float = 7.0
    #: Rebuild missing loops/residues.  Off by default: it invents coordinates.
    add_missing_residues: bool = False
    #: Optional hard cap so a stray huge entry cannot blow up the run.
    max_residues: int = 120


@dataclass
class EnsembleConfig:
    """How conformational ensembles are generated for each protein."""

    #: Seed for every random number generator (reproducibility).
    seed: int = 20240517

    # --- torsional sampling ------------------------------------------------- #
    #: Relative exploration amplitude per level; larger levels rotate further.
    torsion_levels: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
    #: Samples generated per level.
    n_torsion_per_level: int = 12
    #: Rotation angle range for the smallest / largest level, in degrees.
    torsion_min_deg: float = 55.0
    torsion_max_deg: float = 165.0
    #: How many rotatable bonds to rotate per sample (inclusive range).
    moves_min: int = 1
    moves_max: int = 3
    #: Small isotropic jitter (Angstrom) added on top, applied to all atoms.
    #: DEFAULT 0 AND IT SHOULD STAY 0: even 0.04 A of independent Cartesian
    #: noise costs ~0.5 * k * dx^2 per bond, which over 650 bonds adds several
    #: hundred kcal/mol of pure bond-stretching noise that has nothing to do
    #: with conformational energy.  It is kept as an option only so the effect
    #: can be measured.
    cartesian_jitter: float = 0.0
    #: Fraction of torsional samples that are *partially* energy-minimised, to
    #: populate the local-minimum region.  Default 0, and that is a measured
    #: decision: plain torsional rotations about exposed bonds already land near
    #: the minimum (measured median dE ~2 kcal/mol), while each partial
    #: minimisation costs seconds and dominated the whole runtime.  Set to e.g.
    #: 0.15 if you want explicit local-minimum samples and can afford it.
    minimise_fraction: float = 0.0
    #: L-BFGS iterations for partially relaxed samples.
    partial_minimise_iterations: int = 40
    #: L-BFGS iterations for the reference (native) minimum.  This one runs once
    #: per protein and defines the zero of the energy scale, so it is worth
    #: converging reasonably well.
    reference_minimise_iterations: int = 120
    #: Reject any generated frame whose energy exceeds the reference by more
    #: than this many kcal/mol.  Torsional moves can sweep a side chain through
    #: the protein and leave two atoms essentially on top of each other, where
    #: the Lennard-Jones 1/r^12 term reaches 10^9 kcal/mol.  Such geometries are
    #: numerical artefacts, not conformations, and a single one of them would
    #: dominate a squared-error loss.
    max_relative_energy: float = 250.0
    #: Separate, far more generous cap for MD snapshots.  At temperature T a
    #: molecule's potential energy exceeds its minimum by roughly
    #: (1/2) * N_dof * kT through equipartition - for a 650-atom protein at
    #: 300 K that is ~600 kcal/mol, and it is *physics*, not an artefact.  The
    #: cap here exists only to catch a blown-up trajectory.
    md_max_relative_energy: float = 1500.0
    #: When screening rejects frames, keep trying up to
    #: ``n_torsion_per_level * max_attempt_factor`` times per level.
    max_attempt_factor: int = 4

    # --- molecular dynamics ------------------------------------------------- #
    #: Temperatures in Kelvin.  Empty or 0 ps disables MD entirely.
    md_temperatures: tuple[float, ...] = (300.0, 450.0)
    md_ps_per_temperature: float = 0.6
    md_timestep_fs: float = 2.0
    md_save_every_ps: float = 0.2

    # --- experimental conformers ------------------------------------------- #
    #: Also keep models from a deposited NMR ensemble (free real diversity).
    #: Each one needs a short relaxation, so this is capped to keep runtime sane.
    use_nmr_models: bool = True
    max_nmr_models: int = 3
    nmr_minimise_iterations: int = 60


@dataclass
class LabelConfig:
    """Physics engine that produces the ground-truth energies."""

    #: OpenMM force field XML files (relative names resolve inside OpenMM).
    forcefield_files: tuple[str, ...] = ("amber14-all.xml", "implicit/gbn2.xml")
    #: Implicit solvent model: "OBC2", "OBC1", "GBn", "GBn2", "HCT" or None.
    implicit_solvent: str | None = "OBC2"
    solute_dielectric: float = 1.0
    solvent_dielectric: float = 78.5
    #: NoCutoff is exact but O(N^2); a finite cutoff is what production MD uses
    #: and is ~3.5x faster for the Generalised Born term.  1.0-1.2 nm is the
    #: conventional choice for GB implicit-solvent simulations.
    nonbonded_method: str = "CutoffNonPeriodic"
    cutoff_nm: float = 1.0
    #: Constrain bonds involving hydrogen -> allows a 2 fs timestep.
    constraints: str = "HBonds"
    #: OpenMM reports kJ/mol; the paper-trail unit for force fields is kcal/mol.
    energy_unit: str = "kcal/mol"


@dataclass
class FeatureConfig:
    """Graph construction for the neural network."""

    #: Only atoms within this distance (Angstrom) exchange messages.  4.0 A
    #: covers bonds (~1.5 A), angles (~2.4 A) and most 1-4 pairs (~3.0-3.9 A).
    #: Sizing note: cost grows roughly as cutoff^3, so 4.5 A costs ~1.4x what
    #: 4.0 A does - measured, not guessed.
    cutoff: float = 4.0
    #: Number of radial basis functions for the interatomic distance expansion.
    #: The basis is deliberately sparse (3-4 functions are active per distance),
    #: so 32 is plenty and costs ~1/3 less than 48.
    n_rbf: int = 32
    rbf_start: float = 0.0
    rbf_end: float = 5.0
    #: Largest number of neighbours kept per atom (safety valve for memory).
    max_neighbors: int = 32
    #: Append a covalent-bond flag to each edge feature.  Bonded terms dominate
    #: a force-field energy, and the bond graph is information the network would
    #: otherwise have to re-derive from distances alone.
    use_bond_feature: bool = True


@dataclass
class ModelConfig:
    """SchNet-style message-passing network hyper-parameters.

    The number of radial basis functions is *not* here: it must equal
    :attr:`FeatureConfig.n_rbf`, so it lives with the featuriser and the model is
    built to match it.  Keeping one source of truth prevents the classic
    "mat1 and mat2 shapes cannot be multiplied" mismatch.
    """

    kind: str = "schnet"          # "schnet" or "mlp"
    #: Sized for CPU training on a small dataset.  Measured on this machine, one
    #: training step at batch 16 cost 15.6 s with hidden_dim=128/4 interactions
    #: and 4.5 A cutoff, versus ~1.3 s with the values below - a 12x difference
    #: that decides whether the experiment finishes tonight.  With only ~900
    #: training conformations, the larger model would overfit anyway.
    hidden_dim: int = 64
    n_interactions: int = 3
    cutoff: float = 4.0
    n_layers_mlp: int = 3
    dropout: float = 0.0
    #: Predict the energy of every atom then sum (extensive by construction).
    per_atom_output: bool = True


@dataclass
class TrainConfig:
    """Optimisation hyper-parameters."""

    seed: int = 1234
    #: 25 epochs is what this CPU budget allows; early stopping usually fires
    #: first.  The learning curve is saved, so under-training is visible.
    epochs: int = 25
    batch_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    #: Fraction of *proteins* held out for validation and test.  Splitting by
    #: protein (not by frame) is what makes the generalisation claim honest.
    #: With 16 proteins, 0.25/0.25 gives 4 test + 4 validation + 8 training.
    val_fraction: float = 0.25
    test_fraction: float = 0.25
    patience: int = 10
    min_delta: float = 1e-4
    grad_clip: float = 10.0
    num_workers: int = 0
    device: str = "auto"
    #: Cache featurised graphs on disk.  Featurisation costs ~150 ms per frame
    #: and its result never changes, so caching turns an hours-long training run
    #: into minutes.
    cache_graphs: bool = True
    graph_cache_dir: str = "data/processed/graph_cache"
    #: Target transform: "relative" predicts E - E_min(protein).
    target: str = "relative"
    #: "protein" holds out whole proteins (honest); "frame" shuffles frames
    #: (leaky, kept only to demonstrate the leakage effect).
    split_mode: str = "protein"
    #: Loss on the standardised target: "mse" or "huber".
    loss: str = "huber"
    huber_delta: float = 1.0
    #: LR schedule: "plateau" (halve on stagnation) or "cosine".
    scheduler: str = "plateau"
    #: Also learn the per-protein reference energy E_min from structure.
    predict_reference: bool = False
    log_every: int = 10


@dataclass
class Config:
    """Top-level configuration bundle."""

    paths: PathConfig = field(default_factory=PathConfig)
    prepare: PrepareConfig = field(default_factory=PrepareConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    #: PDB IDs to download when ``data/raw`` is empty.
    pdb_ids: tuple[str, ...] = ()

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | os.PathLike) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
        with open(str(path), "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return str(path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            sub = f.type
            if isinstance(value, dict):
                kwargs[f.name] = _build_nested(f.name, value)
            else:
                kwargs[f.name] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Config":
        with open(str(path), "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


_NESTED: dict[str, type] = {
    "paths": PathConfig,
    "prepare": PrepareConfig,
    "ensemble": EnsembleConfig,
    "label": LabelConfig,
    "features": FeatureConfig,
    "model": ModelConfig,
    "train": TrainConfig,
}


def _build_nested(name: str, data: dict[str, Any]):
    cls = _NESTED[name]
    valid = {f.name for f in dataclasses.fields(cls)}
    clean = {k: v for k, v in data.items() if k in valid}
    # JSON has no tuples; convert list -> tuple where the default is a tuple.
    for f in dataclasses.fields(cls):
        if f.name in clean and isinstance(getattr(cls(), f.name), tuple):
            clean[f.name] = tuple(clean[f.name])
    return cls(**clean)


#: Default list of small, well-behaved protein entries used when the user does
#: not provide their own PDB files.  Chosen to be < 120 residues, single-domain,
#: and to include both X-ray and NMR (multi-model) entries.
DEFAULT_PDB_IDS: tuple[str, ...] = (
    "1CRN",  # crambin, 46 aa, X-ray 1.5 A
    "1UBQ",  # ubiquitin, 76 aa, X-ray 1.8 A
    "1D3Z",  # ubiquitin, NMR, 10 models
    "2GB1",  # protein G B1 domain, 56 aa, NMR
    "1L2Y",  # Trp-cage mini protein, 20 aa, NMR, 38 models
    "1VII",  # villin headpiece, 36 aa, NMR
    "5PTI",  # bovine pancreatic trypsin inhibitor, 58 aa, X-ray
    "1PGB",  # protein G B1 domain, X-ray
    "2CI2",  # chymotrypsin inhibitor 2, 65 aa, X-ray
    "1ENH",  # engrailed homeodomain, 54 aa, NMR
    "1BDD",  # protein A B domain, 60 aa, NMR
    "1RIS",  # ribosomal protein S6, 97 aa, X-ray
    "1SHG",  # SH3 domain, 57 aa, X-ray
)
