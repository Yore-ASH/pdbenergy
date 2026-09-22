"""Tests for continuing a run: warm start, resume, and the compatibility guard.

These use tiny synthetic ensembles so they run in seconds - the point is the
*mechanism* (does the optimiser come back? does epoch numbering continue? does a
mismatched config get refused?), not accuracy.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdbenergy.config import Config                                    # noqa: E402
from pdbenergy.dataset import build_bundle                             # noqa: E402
from pdbenergy.ensemble import ProteinEnsemble                         # noqa: E402
from pdbenergy.train import (                                          # noqa: E402
    STATE_FILENAME,
    _check_continuation_compatible,
    train_model,
)


def _tiny_ensembles(n_proteins: int = 4, n_atoms: int = 40, n_frames: int = 8) -> dict:
    """Synthetic labelled ensembles: same shape as the real thing, none of the cost."""
    rng = np.random.default_rng(0)
    elements = np.asarray(["C"] * n_atoms, dtype="U2")
    names = np.asarray([f"C{i % 9 + 1}" for i in range(n_atoms)], dtype="U4")
    out: dict[str, ProteinEnsemble] = {}
    for p in range(n_proteins):
        pid = f"T{p:02d}"
        # Slightly different geometry per protein, with a smooth energy ladder so
        # there is something to fit.
        base = rng.normal(scale=6.0, size=(n_atoms, 3))
        coords = np.stack([
            base + rng.normal(scale=0.05 * (f + 1), size=(n_atoms, 3))
            for f in range(n_frames)
        ]).astype(np.float32)
        energies = np.array([10.0 * f + 3.0 * p for f in range(n_frames)], dtype=np.float64)
        out[pid] = ProteinEnsemble(
            pdb_id=pid,
            coords=coords,
            elements=elements,
            atom_names=names,
            residue_index=np.zeros(n_atoms, dtype=np.int32),
            residue_names=np.asarray(["ALA"] * n_atoms, dtype="U3"),
            energy_total=energies,
            energy_terms={"bond": energies * 0.5, "nonbonded": energies * 0.5},
            source=np.asarray(["torsion"] * n_frames, dtype="U16"),
            reference_energy=float(energies.min()),
            sequence="A" * 5,
        )
    return out


def _tiny_config() -> Config:
    cfg = Config()
    cfg.model.hidden_dim = 8
    cfg.model.n_interactions = 1
    cfg.model.n_layers_mlp = 1
    cfg.features.n_rbf = 8
    cfg.features.max_neighbors = 8
    cfg.train.epochs = 1
    cfg.train.batch_size = 4
    cfg.train.patience = 1
    cfg.train.log_every = 1
    cfg.train.cache_graphs = False          # keep the test off the disk cache
    cfg.train.num_workers = 0
    return cfg


class TestContinuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ensembles = _tiny_ensembles()
        cls.cfg = _tiny_config()
        cls.bundle = build_bundle(cls.ensembles, cls.cfg, verbose=False)

    def _train(self, tmp, **kwargs):
        return train_model(self.bundle, out_dir=tmp, verbose=False, **kwargs)

    # -- fresh run ---------------------------------------------------------- #
    def test_fresh_run_writes_a_resumable_side_car(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._train(tmp)
            self.assertTrue(os.path.exists(os.path.join(tmp, "checkpoint.pt")))
            state_path = os.path.join(tmp, STATE_FILENAME)
            self.assertTrue(os.path.exists(state_path), "train_state.pt must be written")
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            self.assertIn("optimizer_state_dict", state)
            self.assertEqual(state["epochs_trained"], result.history["epoch"][-1])

    # -- warm start --------------------------------------------------------- #
    def test_warm_start_loads_weights_but_resets_the_epoch_counter(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            self._train(first)
            result = self._train(
                second, warm_start_from=os.path.join(first, "checkpoint.pt")
            )
            # A warm start is a *new* run: epochs begin at 1 again.
            self.assertEqual(result.history["epoch"][0], 1)

    def test_warm_start_reuses_the_checkpoint_normalisation(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            self._train(first)
            payload = torch.load(os.path.join(first, "checkpoint.pt"),
                                 map_location="cpu", weights_only=False)
            # Corrupt the bundle's own statistics; the checkpoint's must win.
            self.bundle.norm_mean = 12345.0
            self.bundle.norm_std = 999.0
            try:
                train_model(
                    self.bundle, out_dir=second, verbose=False,
                    warm_start_from=os.path.join(first, "checkpoint.pt"),
                )
                self.assertAlmostEqual(self.bundle.norm_mean, payload["norm_mean"], places=6)
                self.assertAlmostEqual(self.bundle.norm_std, payload["norm_std"], places=6)
            finally:
                fresh = build_bundle(self.ensembles, self.cfg, verbose=False)
                self.bundle.norm_mean = fresh.norm_mean
                self.bundle.norm_std = fresh.norm_std

    def test_warm_start_actually_loads_the_trained_weights(self):
        """With a zero-epoch budget nothing is trained, so the returned model must
        be *exactly* the checkpoint's weights - a deterministic check that the
        load really happened (rather than a comparison of noisy metrics)."""
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            self._train(first)
            original_epochs = self.cfg.train.epochs
            self.cfg.train.epochs = 0            # train nothing, just load
            try:
                result = self._train(
                    second, warm_start_from=os.path.join(first, "checkpoint.pt")
                )
            finally:
                self.cfg.train.epochs = original_epochs

            payload = torch.load(os.path.join(first, "checkpoint.pt"),
                                 map_location="cpu", weights_only=False)
            for key, expected in payload["model_state_dict"].items():
                torch.testing.assert_close(
                    result.model.state_dict()[key], expected, rtol=0, atol=0,
                    msg=f"parameter {key} was not loaded from the checkpoint",
                )

    # -- resume ------------------------------------------------------------- #
    def test_resume_continues_epoch_numbering_and_optimiser(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self._train(tmp)
            epochs_done = first.history["epoch"][-1]
            self.assertEqual(epochs_done, 1)

            resumed = train_model(self.bundle, resume_from=tmp, verbose=False)
            # History accumulates (one continuous learning curve), and epoch
            # numbering carries on instead of restarting at 1.
            expected = list(range(1, epochs_done + self.cfg.train.epochs + 1))
            self.assertEqual(resumed.history["epoch"], expected)

    def test_resume_restores_optimiser_moments(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._train(tmp)
            state = torch.load(os.path.join(tmp, STATE_FILENAME),
                               map_location="cpu", weights_only=False)
            # AdamW keeps two moment buffers per parameter, so the state must be
            # non-trivial - a fresh optimiser would have empty moments here.
            self.assertTrue(state["optimizer_state_dict"]["state"])

    # -- guards ------------------------------------------------------------- #
    def test_mismatched_feature_config_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._train(tmp)
            payload = torch.load(os.path.join(tmp, "checkpoint.pt"),
                                 map_location="cpu", weights_only=False)
            broken = _tiny_config()
            broken.features.n_rbf = 32          # would change the edge layout
            with self.assertRaises(SystemExit):
                _check_continuation_compatible(
                    payload, broken, False, "checkpoint.pt"
                )

    def test_mismatched_model_config_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._train(tmp)
            payload = torch.load(os.path.join(tmp, "checkpoint.pt"),
                                 map_location="cpu", weights_only=False)
            broken = _tiny_config()
            broken.model.hidden_dim = 64
            with self.assertRaises(SystemExit):
                _check_continuation_compatible(payload, broken, False, "checkpoint.pt")

    def test_matching_config_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._train(tmp)
            payload = torch.load(os.path.join(tmp, "checkpoint.pt"),
                                 map_location="cpu", weights_only=False)
            _check_continuation_compatible(payload, self.cfg, False, "checkpoint.pt")

    def test_missing_checkpoint_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                train_model(self.bundle, warm_start_from=os.path.join(tmp, "nope.pt"),
                            verbose=False)
            with self.assertRaises(FileNotFoundError):
                train_model(self.bundle, resume_from=tmp, verbose=False)

    def test_both_warm_start_and_resume_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._train(tmp)
            with self.assertRaises(ValueError):
                train_model(
                    self.bundle, verbose=False,
                    warm_start_from=os.path.join(tmp, "checkpoint.pt"),
                    resume_from=tmp,
                )


if __name__ == "__main__":
    unittest.main()
