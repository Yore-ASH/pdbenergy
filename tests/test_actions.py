"""Tests for the shared action -> command-line mapping.

Both interfaces (PySide6 desktop app and browser GUI) build their child-process
command lines through :mod:`pdbenergy.actions`, so these tests cover the contract
they both rely on.  They also pin the "reproducible command line" property: the
command shown in the UI log must be the one that actually runs.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdbenergy.actions import (                     # noqa: E402
    ActionError,
    available_actions,
    build_job,
)


class TestBuildJob(unittest.TestCase):
    def build(self, action, payload=None, **kw):
        return build_job(action, payload, **kw)

    # -- the basics --------------------------------------------------------- #
    def test_every_advertised_action_is_buildable(self):
        """available_actions() must not list anything that then raises."""
        minimal = {
            "download": {}, "scan": {}, "label": {"ids": ["1CRN"]},
            "dataset": {}, "train": {}, "evaluate": {"run_dir": "outputs/r"},
            "predict": {"paths": ["a.pdb"], "checkpoint": "c.pt"},
            "ablate": {}, "iterate": {}, "measure_leakage": {},
            "estimate_workload": {},
        }
        for action in available_actions():
            with self.subTest(action=action):
                label, module, args = self.build(action, minimal[action])
                self.assertTrue(label)
                self.assertTrue(module)
                self.assertIsInstance(args, list)

    def test_unknown_action_is_rejected(self):
        with self.assertRaises(ActionError):
            self.build("definitely-not-an-action", {})

    def test_cli_actions_target_the_cli_package(self):
        for action, payload in (("scan", {}), ("dataset", {}), ("train", {}),
                                ("ablate", {})):
            with self.subTest(action=action):
                _, module, _ = self.build(action, payload)
                self.assertEqual(module, "pdbenergy.cli")

    def test_iterate_targets_the_iterate_module(self):
        _, module, args = self.build("iterate", {"rounds": 2})
        self.assertEqual(module, "pdbenergy.iterate")
        self.assertIn("--rounds", args)
        self.assertIn("2", args)

    def test_script_actions_use_py_paths_not_m_modules(self):
        """JobManager distinguishes the two by the .py suffix."""
        for action in ("measure_leakage", "estimate_workload"):
            with self.subTest(action=action):
                _, module, args = self.build(action, {})
                self.assertTrue(module.endswith(".py"), module)
                self.assertEqual(args, [])

    # -- required input ----------------------------------------------------- #
    def test_missing_required_input_is_rejected(self):
        for action, payload in (("label", {"ids": []}),
                                ("evaluate", {}),
                                ("predict", {"paths": ["a.pdb"]}),
                                ("predict", {"checkpoint": "c.pt"})):
            with self.subTest(action=action, payload=payload):
                with self.assertRaises(ActionError):
                    self.build(action, payload)

    # -- specific mappings -------------------------------------------------- #
    def test_download_passes_ids_and_force(self):
        _, _, args = self.build("download", {"ids": ["1CRN", "1L2Y"], "force": True})
        self.assertEqual(args[:1], ["download"])
        self.assertIn("--ids", args)
        self.assertIn("1CRN", args)
        self.assertIn("--force", args)

    def test_scan_carries_the_residue_bounds(self):
        _, _, args = self.build("scan", {"max_residues": 200, "min_residues": 20})
        self.assertIn("--no-table", args)
        self.assertIn("200", args)
        self.assertIn("20", args)

    def test_label_counts_ids_in_the_label(self):
        label, _, args = self.build("label", {"ids": ["a", "b", "c"], "threads": 4,
                                              "workers": 2})
        self.assertIn("3", label)
        self.assertIn("--threads", args)
        self.assertIn("4", args)
        self.assertIn("--workers", args)
        self.assertIn("2", args)

    def test_train_defaults_to_a_tag_derived_from_the_model(self):
        _, _, args = self.build("train", {"model": "mlp"})
        self.assertIn("--tag", args)
        self.assertIn("mlp_protein", args)

    def test_train_omits_zero_valued_options(self):
        """0 means 'use the config default', so the flag must be absent entirely."""
        _, _, args = self.build("train", {"epochs": 0, "batch_size": 0,
                                          "learning_rate": 0.0})
        for flag in ("--epochs", "--batch-size", "--learning-rate"):
            self.assertNotIn(flag, args)

    def test_train_passes_positive_options(self):
        _, _, args = self.build("train", {"epochs": 25, "batch_size": 32,
                                          "learning_rate": 0.0005, "hidden_dim": 128})
        self.assertEqual(args[args.index("--epochs") + 1], "25")
        self.assertEqual(args[args.index("--batch-size") + 1], "32")
        self.assertEqual(args[args.index("--hidden-dim") + 1], "128")

    def test_train_forwards_continuation_flags(self):
        _, _, args = self.build("train", {"init_from": "outputs/a/checkpoint.pt",
                                          "no_cache_graphs": True,
                                          "recompute_normalisation": True})
        self.assertIn("--init-from", args)
        self.assertIn("--no-cache-graphs", args)
        self.assertIn("--recompute-normalisation", args)

    def test_predict_writes_into_the_outputs_dir(self):
        _, _, args = self.build("predict", {"paths": ["a.pdb", "b.pdb"],
                                            "checkpoint": "c.pt", "verify": True},
                                outputs_dir="my_outputs")
        self.assertIn("--verify", args)
        self.assertIn("--quiet", args)
        self.assertEqual(args[args.index("--json") + 1],
                         os.path.join("my_outputs", "predictions.json"))
        # Both paths must reach the CLI.
        self.assertIn("a.pdb", args)
        self.assertIn("b.pdb", args)

    def test_iterate_build_new_switches(self):
        _, _, on = self.build("iterate", {"build_new": True})
        _, _, off = self.build("iterate", {"build_new": False})
        self.assertIn("--build-new", on)
        self.assertNotIn("--no-build-new", on)
        self.assertIn("--no-build-new", off)

    def test_iterate_warm_start_switches(self):
        _, _, warm = self.build("iterate", {"warm_start": True})
        _, _, cold = self.build("iterate", {"warm_start": False})
        self.assertNotIn("--no-warm-start", warm)
        self.assertIn("--no-warm-start", cold)

    def test_payload_is_not_mutated(self):
        payload = {"ids": ["1CRN"], "threads": 4}
        snapshot = dict(payload)
        self.build("label", payload)
        self.assertEqual(payload, snapshot)

    def test_none_payload_is_treated_as_empty(self):
        _, _, args = self.build("scan", None)
        self.assertIn("inventory", args)


class TestCommandLineIsReproducible(unittest.TestCase):
    """The command the UI shows must be pasteable into a terminal as-is."""

    def test_round_trip_through_the_cli_parser(self):
        from pdbenergy.cli import build_parser

        parser = build_parser()
        cases = [
            ("scan", {"max_residues": 200, "min_residues": 20}),
            ("dataset", {}),
            ("train", {"model": "schnet", "epochs": 25, "batch_size": 16,
                       "tag": "demo", "split_mode": "protein"}),
            ("evaluate", {"run_dir": os.path.join("outputs", "demo")}),
            ("predict", {"paths": ["a.pdb"], "checkpoint": "c.pt",
                         "max_models": 3, "verify": True}),
            ("ablate", {"epochs": 20, "model": "mlp"}),
        ]
        for action, payload in cases:
            with self.subTest(action=action):
                label, module, args = build_job(action, payload)
                self.assertEqual(module, "pdbenergy.cli")
                # argparse must accept exactly what we built; a missing or
                # renamed flag would raise SystemExit here.
                try:
                    namespace = parser.parse_args(args)
                except SystemExit as exc:               # pragma: no cover
                    self.fail(f"{action}: CLI rejected {args!r} ({exc})")
                self.assertTrue(namespace.command)

    def test_iterate_args_parse_with_the_iterate_parser(self):
        from pdbenergy.iterate import build_parser as iterate_parser

        _, module, args = build_job("iterate", {
            "rounds": 2, "build_limit": 20, "max_residues": 200, "epochs": 20,
            "model": "schnet", "threads": 8, "workers": 1,
            "build_new": True, "warm_start": False,
        })
        self.assertEqual(module, "pdbenergy.iterate")
        namespace = iterate_parser().parse_args(args)
        self.assertEqual(namespace.rounds, 2)
        self.assertEqual(namespace.build_limit, 20)
        self.assertFalse(namespace.warm_start)


if __name__ == "__main__":
    unittest.main()
