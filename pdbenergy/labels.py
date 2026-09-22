"""Ground-truth conformational energy from a real physics force field (OpenMM).

The neural network in this project is a *surrogate model*: it learns to
reproduce the potential energy that a classical molecular-mechanics force field
assigns to a structure.  The force field is therefore the teacher, and this
module is the teacher.

Physics
-------
We use the AMBER14 all-atom protein force field together with the GBn2
(Generalised Born) implicit solvent model:

* bonded terms - harmonic bond stretching, harmonic angle bending, periodic
  torsions, CMAP backbone corrections, harmonic improper torsions;
* non-bonded terms - Lennard-Jones 12-6 van der Waals plus Coulomb
  electrostatics, both optionally screened by a Generalised Born continuum
  solvent instead of thousands of explicit water molecules.

Units and conventions
---------------------
* PDB/our internal coordinates: Angstrom.
* OpenMM internal coordinates: nanometre (1 A = 0.1 nm).
* OpenMM energies: kJ/mol.  Force-field literature uses kcal/mol,
  so every energy leaving this module is divided by 4.184.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import LabelConfig, PrepareConfig
from .pdbio import Structure, protein_only, write_pdb

#: 1 kcal = 4.184 kJ, by definition of the thermochemical calorie.
KJ_PER_KCAL = 4.184

#: Human-readable names for OpenMM force groups.
ENERGY_TERMS: tuple[str, ...] = ("bond", "angle", "torsion", "nonbonded")

#: Which XML file each Generalised Born flavour lives in.
IMPLICIT_SOLVENT_FILES: dict[str, str] = {
    "OBC2": "implicit/obc2.xml",
    "OBC1": "implicit/obc1.xml",
    "GBN": "implicit/gbn.xml",
    "GBN2": "implicit/gbn2.xml",
    "HCT": "implicit/hct.xml",
}


class OpenMMUnavailableError(RuntimeError):
    """Raised when OpenMM/PDBFixer are not installed."""


def _import_openmm():
    try:
        import openmm
        from openmm import app, unit
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise OpenMMUnavailableError(
            "OpenMM and PDBFixer are required for energy labelling.\n"
            "Install them with:  pip install openmm pdbfixer"
        ) from exc
    return openmm, app, unit, PDBFile, PDBFixer


@dataclass
class PreparedSystem:
    """A force-field-ready protein plus the machinery to evaluate it.

    Attributes
    ----------
    topology:
        OpenMM topology of the *prepared* all-atom system (hydrogens included).
    system:
        OpenMM System holding the force-field parameters.
    heavy_indices:
        Indices into the prepared atom order that are heavy (non-hydrogen)
        atoms.  The neural network only ever sees these.
    heavy_elements / heavy_names:
        Per-heavy-atom element symbol and PDB atom name, in prepared order.
    positions_nm:
        Current positions in nanometres, shape (N_prepared, 3).
    reference_energy_kcal:
        Energy of the energy-minimised structure; the zero of the relative
        energy scale used as the learning target.
    """

    pdb_id: str
    topology: object
    system: object
    heavy_indices: np.ndarray
    heavy_elements: list[str]
    heavy_names: list[str]
    heavy_residue_index: np.ndarray
    heavy_residue_names: list[str]
    positions_nm: np.ndarray
    reference_energy_kcal: float | None = None
    n_atoms_prepared: int = 0
    n_atoms_heavy: int = 0

    def heavy_coords_angstrom(self, positions_nm: np.ndarray | None = None) -> np.ndarray:
        """Extract the heavy-atom coordinates in Angstrom from a full frame."""
        pos = self.positions_nm if positions_nm is None else positions_nm
        return np.asarray(pos, dtype=np.float64)[self.heavy_indices] * 10.0

    def heavy_structure(self, positions_nm: np.ndarray | None = None) -> Structure:
        """Build a :class:`Structure` (heavy atoms only) from a frame."""
        from .pdbio import Atom

        coords = self.heavy_coords_angstrom(positions_nm)
        atoms = []
        for i, (elem, name) in enumerate(zip(self.heavy_elements, self.heavy_names)):
            atoms.append(
                Atom(
                    serial=i + 1,
                    name=name,
                    altloc="",
                    resname=self.heavy_residue_names[i],
                    chain="A",
                    resseq=int(self.heavy_residue_index[i]) + 1,
                    icode="",
                    x=float(coords[i, 0]), y=float(coords[i, 1]), z=float(coords[i, 2]),
                    occupancy=1.0, bfactor=0.0, element=elem,
                )
            )
        return Structure(atoms, self.pdb_id, 1)


class OpenMMEnergyEngine:
    """Prepare proteins and evaluate/minimise/propagate their energy.

    A single engine object is cheap; the expensive part is
    :meth:`prepare`, which builds the OpenMM ``System``.  Because OpenMM
    ``Context`` objects are not thread-safe, each worker process should own its
    own engine.
    """

    def __init__(
        self,
        cfg: LabelConfig | None = None,
        prepare_cfg: PrepareConfig | None = None,
        *,
        threads: int = 1,
        verbose: bool = False,
    ):
        self.cfg = cfg or LabelConfig()
        self.prepare_cfg = prepare_cfg or PrepareConfig()
        self.threads = max(1, int(threads))
        self.verbose = verbose
        self._openmm = None

    # -- lazy OpenMM access ------------------------------------------------- #
    @property
    def openmm(self):
        if self._openmm is None:
            self._openmm = _import_openmm()
        return self._openmm

    def _nonbonded_method(self):
        _, app, _, _, _ = self.openmm
        name = self.cfg.nonbonded_method
        method = getattr(app, name)
        return method

    # -- preparation -------------------------------------------------------- #
    def protonate(
        self, structure: Structure, pdb_id: str, *, workdir: str | None = None
    ) -> tuple[object, np.ndarray]:
        """Clean and protonate a structure *without* building a force field.

        Returns ``(topology, positions_nm)``.  This is the cheap half of
        :meth:`prepare`: it needs PDBFixer but not a parameterised ``System``.
        That matters at inference time - the neural network needs atoms and
        coordinates, not force-field parameters, so prediction avoids the
        expensive parameter-assignment step entirely.
        """
        openmm, app, unit, PDBFile, PDBFixer = self.openmm
        pcfg = self.prepare_cfg

        clean = protein_only(structure, keep_hydrogens=False)
        if len(clean) == 0:
            raise ValueError(f"{pdb_id}: no standard amino-acid atoms found")

        workdir = workdir or tempfile.mkdtemp(prefix="pdbenergy_")
        os.makedirs(workdir, exist_ok=True)
        src = os.path.join(workdir, f"{pdb_id}_clean.pdb")
        write_pdb(clean, src, remark=f"cleaned input for {pdb_id}")

        fixer = PDBFixer(filename=src)
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingResidues()
        if not pcfg.add_missing_residues:
            # Rebuilding missing loops fabricates coordinates; keep it opt-in.
            fixer.missingResidues = {}
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        if fixer.missingAtoms or fixer.missingTerminals:
            fixer.addMissingAtoms()
        if pcfg.add_hydrogens:
            fixer.addMissingHydrogens(pcfg.ph)
        positions = np.array(fixer.positions.value_in_unit(unit.nanometer), dtype=np.float64)
        return fixer.topology, positions

    def prepare(self, structure: Structure, pdb_id: str, *, workdir: str | None = None) -> PreparedSystem:
        """Clean, repair, protonate and parameterise one protein.

        Steps (this is the standard "system preparation" workflow in
        biomolecular simulation):

        1. keep only standard amino acids (drop waters, ions, ligands, altlocs);
        2. let PDBFixer find non-standard residues and missing heavy atoms;
        3. rebuild the missing heavy atoms and add hydrogens at a target pH;
        4. build the OpenMM ``System`` (parameter assignment);
        5. energy-minimise: this removes the strain that crystal/NMR deposition
           or hydrogen placement introduced and defines the energy reference.
        """
        openmm, app, unit, PDBFile, PDBFixer = self.openmm
        topology, positions = self.protonate(structure, pdb_id, workdir=workdir)

        system = self._create_system(topology)

        # -- record the heavy-atom view once, in prepared atom order -------- #
        heavy_indices, elements, names = [], [], []
        residue_index, residue_names = [], []
        for i, atom in enumerate(topology.atoms()):
            sym = atom.element.symbol if atom.element is not None else "X"
            if sym.upper() in ("H", "D"):
                continue
            heavy_indices.append(i)
            elements.append(sym.upper())
            names.append(atom.name)
            residue_index.append(atom.residue.index)
            residue_names.append(atom.residue.name)

        prepared = PreparedSystem(
            pdb_id=pdb_id,
            topology=topology,
            system=system,
            heavy_indices=np.asarray(heavy_indices, dtype=np.int64),
            heavy_elements=elements,
            heavy_names=names,
            heavy_residue_index=np.asarray(residue_index, dtype=np.int64),
            heavy_residue_names=residue_names,
            positions_nm=positions,
            n_atoms_prepared=system.getNumParticles(),
            n_atoms_heavy=len(heavy_indices),
        )
        return prepared

    def _create_system(self, topology):
        """Assign force-field parameters and tag forces with energy groups.

        Note on implicit solvent
        ------------------------
        In OpenMM 8.x the Generalised Born model is *not* selected by an
        ``implicitSolvent=`` argument.  It is selected by including the matching
        XML file (``implicit/gbn2.xml``) in the ``ForceField``: that file runs an
        embedded ``<Script>`` which builds a ``CustomGBForce`` using the
        non-bonded charges and the ``soluteDielectric`` / ``solventDielectric``
        / ``sasaMethod`` keyword arguments.  Passing ``implicitSolvent`` raises
        "argument was specified but never used".
        """
        openmm, app, unit, _, _ = self.openmm
        cfg = self.cfg

        files = list(cfg.forcefield_files)
        if cfg.implicit_solvent and not any("implicit" in f for f in files):
            mapped = IMPLICIT_SOLVENT_FILES.get(cfg.implicit_solvent.upper())
            if mapped is None:
                raise ValueError(
                    f"unknown implicit solvent {cfg.implicit_solvent!r}; "
                    f"choose one of {sorted(IMPLICIT_SOLVENT_FILES)} or set implicit_solvent=None"
                )
            files.append(mapped)

        forcefield = app.ForceField(*files)
        kwargs: dict = {
            "nonbondedMethod": self._nonbonded_method(),
            "constraints": getattr(app, cfg.constraints),
            "rigidWater": False,
            "removeCMMotion": False,
        }
        if cfg.nonbonded_method.lower().startswith("cutoff"):
            kwargs["nonbondedCutoff"] = cfg.cutoff_nm * unit.nanometer
        if cfg.implicit_solvent:
            # Consumed by the <Script> inside implicit/*.xml.
            kwargs["soluteDielectric"] = cfg.solute_dielectric
            kwargs["solventDielectric"] = cfg.solvent_dielectric
        system = forcefield.createSystem(topology, **kwargs)

        # Energy decomposition: one force group per physically meaningful term,
        # so `State(groups={i})` reports that term alone.
        for force in system.getForces():
            cls = type(force).__name__
            if cls == "HarmonicBondForce":
                force.setForceGroup(0)
            elif cls == "HarmonicAngleForce":
                force.setForceGroup(1)
            elif cls in ("PeriodicTorsionForce", "RBTorsionForce"):
                force.setForceGroup(2)
            elif cls in ("NonbondedForce", "CustomNonbondedForce"):
                force.setForceGroup(3)
            elif cls in ("CMMotionRemover",):
                force.setForceGroup(0)
            else:
                # Anything else (e.g. CMAP, AMOEBA multipoles) rides with torsions.
                force.setForceGroup(2)
        return system

    # -- context helpers ---------------------------------------------------- #
    def _platform(self):
        openmm, app, unit, _, _ = self.openmm
        platform = openmm.Platform.getPlatformByName("CPU")
        props = {"Threads": str(self.threads)}
        return platform, props

    def make_context(self, prepared: PreparedSystem, *, for_md: bool = False, temperature_k: float = 300.0):
        """Create an OpenMM Context bound to a fresh copy of the System."""
        openmm, app, unit, _, _ = self.openmm
        # OpenMM Contexts each hold their own copy of the particles, so a single
        # System object can safely back many Contexts.  Never share a *Context*
        # between threads, though: contexts are stateful and not thread-safe.
        system = prepared.system
        if for_md:
            integrator = openmm.LangevinMiddleIntegrator(
                temperature_k * unit.kelvin,
                1.0 / unit.picosecond,
                2.0 * unit.femtoseconds,
            )
        else:
            integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
        platform, props = self._platform()
        context = openmm.Context(system, integrator, platform, props)
        context.setPositions(prepared.positions_nm * unit.nanometer)
        return context, integrator

    # -- energy ------------------------------------------------------------- #
    def energy_kcal(self, context) -> float:
        """Total potential energy of the context's current positions."""
        openmm, app, unit, _, _ = self.openmm
        state = context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    def energy_terms_kcal(self, context) -> dict[str, float]:
        """Per-term energy decomposition using the force groups set above."""
        openmm, app, unit, _, _ = self.openmm
        out: dict[str, float] = {}
        for group, term in enumerate(ENERGY_TERMS):
            state = context.getState(getEnergy=True, groups={group})
            out[term] = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        return out

    # -- minimisation ------------------------------------------------------- #
    def minimise(self, context, max_iterations: int = 250) -> float:
        """L-BFGS energy minimisation in place; returns the final energy."""
        openmm, app, unit, _, _ = self.openmm
        openmm.LocalEnergyMinimizer.minimize(context, 1.0, max_iterations)
        return self.energy_kcal(context)

    # -- molecular dynamics ------------------------------------------------- #
    def run_md(
        self,
        prepared: PreparedSystem,
        *,
        temperature_k: float,
        n_steps: int,
        save_every: int,
        seed: int | None = None,
        start_positions_angstrom: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Run Langevin MD and return snapshots as (N_prepared, 3) Angstrom arrays.

        Langevin dynamics adds friction and a random force, which together act
        as a thermostat: the system samples the canonical (NVT) ensemble at
        ``temperature_k``.  Snapshots therefore represent *thermally accessible*
        conformations, i.e. the physically relevant part of the energy landscape.

        Start from an energy-minimised structure (``start_positions_angstrom``)
        unless you deliberately want to watch a strained structure relax: a
        deposited PDB with hydrogens just added sits hundreds of kcal/mol above
        the local minimum, and a 1 ps run is nowhere near long enough to relax
        that away.
        """
        openmm, app, unit, _, _ = self.openmm
        context, integrator = self.make_context(prepared, for_md=True, temperature_k=temperature_k)
        if start_positions_angstrom is not None:
            self.set_positions(context, start_positions_angstrom)
        if seed is not None:
            integrator.setRandomNumberSeed(int(seed))
        context.setVelocitiesToTemperature(temperature_k * unit.kelvin, int(seed or 0))

        frames: list[np.ndarray] = []
        for step in range(1, n_steps + 1):
            integrator.step(1)
            if step % save_every == 0:
                state = context.getState(getPositions=True)
                pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
                frames.append(np.array(pos, dtype=np.float64))
        del context, integrator
        return frames

    def snapshot_energy_kcal(
        self, prepared: PreparedSystem, positions_angstrom: np.ndarray, *, terms: bool = False
    ) -> tuple[float, dict[str, float] | None]:
        """Single-point energy of an arbitrary all-atom frame."""
        context, _ = self.make_context(prepared)
        self.set_positions(context, positions_angstrom)
        energy = self.energy_kcal(context)
        term_values = self.energy_terms_kcal(context) if terms else None
        del context
        return energy, term_values

    def set_positions(self, context, positions_angstrom: np.ndarray) -> None:
        openmm, app, unit, _, _ = self.openmm
        arr = np.asarray(positions_angstrom, dtype=np.float64)
        if arr.shape[0] != context.getSystem().getNumParticles():
            raise ValueError(
                f"expected {context.getSystem().getNumParticles()} atoms, got {arr.shape[0]}"
            )
        context.setPositions(arr / 10.0)


def engine_available() -> bool:
    """True when OpenMM + PDBFixer can be imported."""
    try:
        _import_openmm()
        return True
    except OpenMMUnavailableError:
        return False
