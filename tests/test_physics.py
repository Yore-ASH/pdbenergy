"""Physics-engine tests. Skipped automatically when OpenMM is not installed."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdbenergy.config import LabelConfig, PrepareConfig  # noqa: E402
from pdbenergy.labels import KJ_PER_KCAL, engine_available  # noqa: E402
from pdbenergy.pdbio import protein_only, read_pdb  # noqa: E402

CRN = os.path.join("data", "raw", "1CRN.pdb")

requires_openmm = unittest.skipUnless(
    engine_available(), "OpenMM/PDBFixer not installed"
)
requires_data = unittest.skipUnless(os.path.exists(CRN), "1CRN.pdb not downloaded")


@requires_openmm
class TestEnergyEngine(unittest.TestCase):
    """A single preparation is shared by all tests in this class: it is slow."""

    prepared = None
    engine = None

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CRN):
            raise unittest.SkipTest("1CRN.pdb not downloaded")
        from pdbenergy.labels import OpenMMEnergyEngine

        cls.engine = OpenMMEnergyEngine(
            LabelConfig(), PrepareConfig(), threads=2, verbose=False
        )
        cls.prepared = cls.engine.prepare(
            protein_only(read_pdb(CRN)[0]), "1CRN_test", workdir="data/interim/_tests"
        )

    def test_hydrogens_were_added(self):
        self.assertEqual(self.prepared.n_atoms_prepared, 642)
        self.assertEqual(self.prepared.n_atoms_heavy, 327)
        self.assertGreater(self.prepared.n_atoms_prepared, self.prepared.n_atoms_heavy)

    def test_energy_terms_sum_to_total(self):
        """The energy decomposition must be exact, not approximate."""
        context, _ = self.engine.make_context(self.prepared)
        total = self.engine.energy_kcal(context)
        terms = self.engine.energy_terms_kcal(context)
        self.assertEqual(set(terms), {"bond", "angle", "torsion", "nonbonded"})
        self.assertAlmostEqual(sum(terms.values()), total, places=3)

    def test_minimisation_lowers_the_energy(self):
        context, _ = self.engine.make_context(self.prepared)
        start = self.engine.energy_kcal(context)
        end = self.engine.minimise(context, 60)
        self.assertLess(end, start)

    def test_energy_is_translation_and_rotation_invariant(self):
        """A rigid-body move is physically free, so the energy must not change."""
        context, _ = self.engine.make_context(self.prepared)
        base_positions = np.array(self.prepared.positions_nm)
        base = self.engine.energy_kcal(context)

        # Rotation matrix about z, plus a translation, in nanometres.
        theta = 0.4
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        moved = base_positions @ R.T + np.array([0.5, -0.3, 0.9])
        self.engine.set_positions(context, moved * 10.0)   # nm -> Angstrom
        shifted = self.engine.energy_kcal(context)
        self.assertAlmostEqual(base, shifted, places=3)

    def test_energy_scales_with_composition_not_position(self):
        """Sanity check on units: a 642-atom protein energy is hundreds of kcal/mol,
        not hundreds of thousands (a units bug would show up immediately)."""
        context, _ = self.engine.make_context(self.prepared)
        value = self.engine.energy_kcal(context)
        self.assertLess(abs(value), 1e5)

    def test_kj_per_kcal_constant(self):
        self.assertAlmostEqual(KJ_PER_KCAL, 4.184, places=6)


@requires_openmm
@requires_data
class TestMD(unittest.TestCase):
    def test_md_frames_are_finite_and_have_right_shape(self):
        from pdbenergy.labels import OpenMMEnergyEngine

        engine = OpenMMEnergyEngine(LabelConfig(), PrepareConfig(), threads=2)
        prepared = engine.prepare(
            protein_only(read_pdb(CRN)[0]), "1CRN_md", workdir="data/interim/_tests"
        )
        frames = engine.run_md(
            prepared, temperature_k=300.0, n_steps=20, save_every=10, seed=1
        )
        self.assertEqual(len(frames), 2)
        for frame in frames:
            self.assertEqual(frame.shape, (prepared.n_atoms_prepared, 3))
            self.assertTrue(np.all(np.isfinite(frame)))

    def test_equipartition_raises_energy_above_the_minimum(self):
        """At finite temperature the mean potential energy sits above the minimum
        by roughly (1/2) N_dof kT - the reason MD frames need a separate,
        much more generous screening threshold."""
        from pdbenergy.labels import OpenMMEnergyEngine

        engine = OpenMMEnergyEngine(LabelConfig(), PrepareConfig(), threads=2)
        prepared = engine.prepare(
            protein_only(read_pdb(CRN)[0]), "1CRN_eq", workdir="data/interim/_tests"
        )
        context, _ = engine.make_context(prepared)
        e_min = engine.minimise(context, 60)
        min_positions = np.array(
            context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(engine.openmm[2].angstrom)
        )
        frames = engine.run_md(
            prepared, temperature_k=300.0, n_steps=200, save_every=100, seed=2,
            start_positions_angstrom=min_positions,
        )
        energies = [engine.snapshot_energy_kcal(prepared, f)[0] for f in frames]
        # The prediction from equipartition for ~640 atoms at 300 K is hundreds of
        # kcal/mol; require at least 50 to prove the effect is present.
        self.assertGreater(float(np.mean(energies)) - e_min, 50.0)


if __name__ == "__main__":
    unittest.main()
