"""Unit and property tests for the ML side: featurisation, models, splits."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdbenergy.config import Config, FeatureConfig, ModelConfig  # noqa: E402
from pdbenergy.dataset import split_proteins  # noqa: E402
from pdbenergy.ensemble import (  # noqa: E402
    ProteinEnsemble,
    adjacency,
    rotation_matrix,
    rotate_torsion,
)
from pdbenergy.features import (  # noqa: E402
    atomic_numbers,
    bond_flags_from_distances,
    collate_graphs,
    frame_to_graph,
    global_descriptors,
    gaussian_rbf,
    pairwise_distances,
)
from pdbenergy.models import (  # noqa: E402
    GaussianRBFExpansion,
    build_model,
    count_parameters,
    edge_dims_for,
    scatter_sum,
)

FEATURE_CFG = FeatureConfig()


def _fake_water():
    """A tiny 'molecule': 4 atoms with a reproducible geometry."""
    coords = np.array([
        [0.000, 0.000, 0.000],   # O
        [0.957, 0.000, 0.000],   # H
        [-0.240, 0.927, 0.000],  # H
        [3.500, 0.000, 0.000],   # a distant C
    ])
    return coords, ["O", "H", "H", "C"]


class TestDistances(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        coords = rng.normal(size=(12, 3))
        got = pairwise_distances(coords)
        want = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        np.fill_diagonal(want, np.inf)
        np.testing.assert_allclose(got, want, atol=1e-8)

    def test_diagonal_is_infinite(self):
        coords, _ = _fake_water()
        d = pairwise_distances(coords)
        self.assertTrue(np.all(np.isinf(np.diag(d))))

    def test_symmetric(self):
        coords, _ = _fake_water()
        d = pairwise_distances(coords)
        np.testing.assert_allclose(d, d.T, equal_nan=True)


class TestRBF(unittest.TestCase):
    def test_shape_and_range(self):
        basis = gaussian_rbf(np.array([0.0, 1.0, 4.9]), 48, 0.0, 5.0)
        self.assertEqual(basis.shape, (3, 48))
        # Distant Gaussians underflow to exactly 0.0, which is expected and
        # harmless - the basis is deliberately sparse.
        self.assertTrue(np.all(basis >= 0.0))
        self.assertTrue(np.all(basis <= 1.0 + 1e-6))
        self.assertTrue(np.all(basis.sum(axis=1) > 0.0))
        # Each distance must light up only a few nearby basis functions.
        self.assertLess(float((basis[1] > 1e-3).sum()), 12)

    def test_torch_expansion_has_cutoff_envelope(self):
        expansion = GaussianRBFExpansion(16, cutoff=4.0, rbf_start=0.0, rbf_end=4.0)
        out = expansion(torch.tensor([1.0, 4.0, 10.0]))
        self.assertEqual(out.shape, (3, 16))
        # Exactly at the cutoff the envelope is zero; beyond it the message must
        # vanish, otherwise the predicted energy is discontinuous.
        self.assertAlmostEqual(float(out[1].abs().sum()), 0.0, places=6)
        self.assertAlmostEqual(float(out[2].abs().sum()), 0.0, places=6)
        self.assertGreater(float(out[0].abs().sum()), 0.0)


class TestBondFlags(unittest.TestCase):
    def test_oh_bond_is_flagged_but_distant_atom_is_not(self):
        coords, elements = _fake_water()
        flags = bond_flags_from_distances(pairwise_distances(coords), elements)
        self.assertEqual(flags[0, 1], 1.0)      # O-H at 0.957 A
        self.assertEqual(flags[0, 2], 1.0)      # O-H
        self.assertEqual(flags[0, 3], 0.0)      # O...C at 3.5 A
        self.assertEqual(flags[1, 2], 0.0)      # H...H


class TestGraphConstruction(unittest.TestCase):
    def test_edge_count_matches_distance_cutoff(self):
        coords, elements = _fake_water()
        graph = frame_to_graph(coords, elements, FEATURE_CFG)
        d = pairwise_distances(coords)
        expected = int((d < FEATURE_CFG.cutoff).sum())
        self.assertEqual(graph.n_edges, expected)
        self.assertEqual(graph.edge_attr.shape[0], expected)

    def test_no_self_loops(self):
        coords, elements = _fake_water()
        graph = frame_to_graph(coords, elements, FEATURE_CFG)
        self.assertTrue(np.all(graph.edge_index[0] != graph.edge_index[1]))

    def test_edges_are_within_cutoff_and_distance_column_is_correct(self):
        coords, elements = _fake_water()
        graph = frame_to_graph(coords, elements, FEATURE_CFG)
        recv, send = graph.edge_index[0], graph.edge_index[1]
        dist = graph.edge_attr[:, 0]
        self.assertTrue(np.all(dist < FEATURE_CFG.cutoff))
        ref = np.linalg.norm(coords[recv] - coords[send], axis=1)
        np.testing.assert_allclose(dist, ref, atol=1e-5)

    def test_atomic_numbers(self):
        _, elements = _fake_water()
        np.testing.assert_array_equal(atomic_numbers(elements), [8, 1, 1, 6])
        np.testing.assert_array_equal(atomic_numbers(["Xx"]), [0])

    def test_rotation_and_translation_leave_features_invariant(self):
        """The edge-distance features must not change under rigid motion."""
        coords, elements = _fake_water()
        g1 = frame_to_graph(coords, elements, FEATURE_CFG)
        R = rotation_matrix(np.array([1.0, 2.0, 0.5]), 0.7)
        moved = coords @ R.T + np.array([10.0, -3.0, 7.0])
        g2 = frame_to_graph(moved, elements, FEATURE_CFG)
        self.assertEqual(g1.n_edges, g2.n_edges)
        np.testing.assert_allclose(
            np.sort(g1.edge_attr[:, 0]), np.sort(g2.edge_attr[:, 0]), atol=1e-4
        )


class TestDescriptors(unittest.TestCase):
    def test_shape_and_finiteness(self):
        coords, elements = _fake_water()
        x = global_descriptors(coords, elements)
        self.assertEqual(x.shape, (24,))
        self.assertTrue(np.all(np.isfinite(x)))

    def test_rotation_invariance(self):
        coords, elements = _fake_water()
        R = rotation_matrix(np.array([0.3, 1.0, -0.2]), 1.1)
        a = global_descriptors(coords, elements)
        b = global_descriptors(coords @ R.T + 5.0, elements)
        np.testing.assert_allclose(a, b, atol=1e-4)

    def test_empty_input(self):
        self.assertEqual(global_descriptors(np.zeros((0, 3)), []).shape, (24,))


class TestScatter(unittest.TestCase):
    def test_matches_looped_sum(self):
        src = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        index = torch.tensor([0, 0, 1, 1, 2, 2])
        got = scatter_sum(src, index, 3)
        want = torch.stack([src[0] + src[1], src[2] + src[3], src[4] + src[5]])
        torch.testing.assert_close(got, want)

    def test_is_permutation_invariant(self):
        """The heart of permutation invariance: shuffling atoms changes nothing."""
        src = torch.randn(9, 3)
        index = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        base = scatter_sum(src, index, 3)
        perm = torch.randperm(9)
        shuffled = scatter_sum(src[perm], index[perm], 3)
        torch.testing.assert_close(base, shuffled, rtol=1e-5, atol=1e-5)


class TestGraphCollation(unittest.TestCase):
    def test_batch_offsets_and_grouping(self):
        coords, elements = _fake_water()
        g = frame_to_graph(coords, elements, FEATURE_CFG)
        item = {
            "z": torch.from_numpy(g.z),
            "pos": torch.from_numpy(g.pos),
            "edge_index": torch.from_numpy(g.edge_index),
            "edge_attr": torch.from_numpy(g.edge_attr),
            "y": torch.tensor(1.0),
            "protein": "X",
            "frame": 0,
            "source": "toy",
        }
        batch = collate_graphs([item, item])
        self.assertEqual(batch["n_graphs"], 2)
        self.assertEqual(batch["z"].shape[0], 8)
        self.assertEqual(batch["y"].shape, (2,))
        # Second graph's edges must be offset by the first graph's atom count.
        second = batch["edge_index"][:, g.n_edges:]
        self.assertGreaterEqual(int(second.min()), 4)
        # batch vector must partition the atoms into two equal groups
        self.assertEqual(int((batch["batch"] == 0).sum()), 4)
        self.assertEqual(int((batch["batch"] == 1).sum()), 4)


class TestModel(unittest.TestCase):
    def _batch(self, n_graphs=2, seed=0):
        rng = np.random.default_rng(seed)
        items = []
        for _ in range(n_graphs):
            coords = rng.normal(scale=1.5, size=(7, 3))
            elements = ["C", "C", "N", "O", "H", "H", "S"]
            g = frame_to_graph(coords, elements, FEATURE_CFG)
            items.append({
                "z": torch.from_numpy(g.z), "pos": torch.from_numpy(g.pos),
                "edge_index": torch.from_numpy(g.edge_index),
                "edge_attr": torch.from_numpy(g.edge_attr),
                "y": torch.tensor(0.0), "protein": "X", "frame": 0, "source": "toy",
            })
        return collate_graphs(items)

    def test_forward_shape(self):
        model = build_model(ModelConfig(kind="schnet", hidden_dim=16, n_interactions=2),
                            **edge_dims_for(FEATURE_CFG))
        batch = self._batch()
        out = model(
            z=batch["z"], edge_index=batch["edge_index"], edge_attr=batch["edge_attr"],
            batch=batch["batch"], n_graphs=batch["n_graphs"],
        )
        self.assertEqual(out.shape, (2,))
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_edge_dims_are_derived_from_features(self):
        cfg = FeatureConfig(n_rbf=32, use_bond_feature=True)
        kwargs = edge_dims_for(cfg)
        self.assertEqual(kwargs["n_rbf"], 32)
        self.assertEqual(kwargs["extra_edge_dim"], 1)

    def test_permutation_invariance_of_prediction(self):
        """Reordering atoms must not change the predicted energy."""
        torch.manual_seed(0)
        model = build_model(ModelConfig(kind="schnet", hidden_dim=16, n_interactions=2),
                            **edge_dims_for(FEATURE_CFG)).eval()
        coords = np.random.default_rng(3).normal(scale=1.3, size=(8, 3))
        elements = ["C", "N", "O", "C", "C", "S", "H", "H"]

        def predict(coords_, elements_):
            g = frame_to_graph(coords_, elements_, FEATURE_CFG)
            with torch.no_grad():
                return float(model(
                    z=torch.from_numpy(g.z),
                    edge_index=torch.from_numpy(g.edge_index),
                    edge_attr=torch.from_numpy(g.edge_attr),
                    batch=torch.zeros(g.n_atoms, dtype=torch.long),
                    n_graphs=1,
                )[0])

        base = predict(coords, elements)
        perm = np.random.default_rng(7).permutation(len(coords))
        shuffled = predict(coords[perm], [elements[i] for i in perm])
        self.assertAlmostEqual(base, shuffled, places=4)

    def test_rotation_invariance_of_prediction(self):
        torch.manual_seed(0)
        model = build_model(ModelConfig(kind="schnet", hidden_dim=16, n_interactions=2),
                            **edge_dims_for(FEATURE_CFG)).eval()
        coords = np.random.default_rng(4).normal(scale=1.4, size=(9, 3))
        elements = ["C", "N", "O", "C", "C", "S", "H", "H", "H"]
        g1 = frame_to_graph(coords, elements, FEATURE_CFG)
        R = rotation_matrix(np.array([0.2, 1.0, 0.4]), 0.9)
        g2 = frame_to_graph(coords @ R.T + np.array([4.0, 4.0, -2.0]), elements, FEATURE_CFG)
        with torch.no_grad():
            e1 = model(z=torch.from_numpy(g1.z), edge_index=torch.from_numpy(g1.edge_index),
                       edge_attr=torch.from_numpy(g1.edge_attr),
                       batch=torch.zeros(g1.n_atoms, dtype=torch.long), n_graphs=1)
            e2 = model(z=torch.from_numpy(g2.z), edge_index=torch.from_numpy(g2.edge_index),
                       edge_attr=torch.from_numpy(g2.edge_attr),
                       batch=torch.zeros(g2.n_atoms, dtype=torch.long), n_graphs=1)
        self.assertAlmostEqual(float(e1[0]), float(e2[0]), places=3)

    def test_mlp_baseline_forward(self):
        model = build_model(ModelConfig(kind="mlp", hidden_dim=16, n_layers_mlp=2))
        out = model(torch.randn(5, 24))
        self.assertEqual(out.shape, (5,))

    def test_parameter_count_positive(self):
        model = build_model(ModelConfig(kind="schnet", hidden_dim=16, n_interactions=2),
                            **edge_dims_for(FEATURE_CFG))
        self.assertGreater(count_parameters(model), 0)


class TestTorsionRotation(unittest.TestCase):
    def test_rotation_preserves_all_bond_lengths_and_angles(self):
        """The whole point of torsional moves: only the dihedral changes."""
        # Build a 4-atom chain A-B-C-D plus one branch atom on C.
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.4, 0.0],
            [1.5, 2.5, 1.0],
            [3.5, 1.6, 0.0],
        ])
        bonds = [(0, 1), (1, 2), (2, 3), (2, 4)]
        adj = adjacency(5, bonds)
        rotated = rotate_torsion(coords, adj, 1, 2, np.deg2rad(80.0))

        # Bond lengths unchanged.
        for i, j in bonds:
            before = np.linalg.norm(coords[i] - coords[j])
            after = np.linalg.norm(rotated[i] - rotated[j])
            self.assertAlmostEqual(before, after, places=8)

        # All bond angles unchanged.
        for a, b, c in [(0, 1, 2), (1, 2, 3), (1, 2, 4), (3, 2, 4)]:
            v1, v2 = coords[a] - coords[b], coords[c] - coords[b]
            w1, w2 = rotated[a] - rotated[b], rotated[c] - rotated[b]
            cos1 = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos2 = w1 @ w2 / (np.linalg.norm(w1) * np.linalg.norm(w2))
            self.assertAlmostEqual(cos1, cos2, places=8)

        # The axis atoms themselves must not move.
        np.testing.assert_allclose(rotated[1], coords[1], atol=1e-10)
        np.testing.assert_allclose(rotated[2], coords[2], atol=1e-10)
        # ...but the downstream atoms must.
        self.assertGreater(np.linalg.norm(rotated[3] - coords[3]), 0.1)

    def test_zero_rotation_is_identity(self):
        coords = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.5, 1.0, 0.0]])
        adj = adjacency(3, [(0, 1), (1, 2)])
        out = rotate_torsion(coords, adj, 0, 1, 0.0)
        np.testing.assert_allclose(out, coords, atol=1e-10)

    def test_rotation_matrix_is_orthonormal(self):
        R = rotation_matrix(np.array([0.3, -1.2, 0.7]), 0.63)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=10)


class TestSplits(unittest.TestCase):
    def test_disjoint_and_complete(self):
        proteins = [f"P{i:02d}" for i in range(12)]
        splits = split_proteins(proteins, seed=1, val_fraction=0.2, test_fraction=0.2)
        all_ids = [p for ids in splits.values() for p in ids]
        self.assertEqual(sorted(all_ids), sorted(proteins))   # nothing lost
        self.assertEqual(len(all_ids), len(set(all_ids)))     # nothing duplicated
        self.assertTrue(splits["train"] and splits["val"] and splits["test"])

    def test_deterministic(self):
        proteins = [f"P{i}" for i in range(10)]
        a = split_proteins(proteins, seed=42)
        b = split_proteins(proteins, seed=42)
        self.assertEqual(a, b)
        c = split_proteins(proteins, seed=43)
        self.assertNotEqual(a["test"], c["test"])

    def test_too_few_proteins_degrades_gracefully(self):
        splits = split_proteins(["A", "B"], seed=0)
        self.assertEqual(sorted(splits["train"]), ["A", "B"])


class TestEnsembleRoundTrip(unittest.TestCase):
    def test_save_load(self):
        import tempfile

        ens = ProteinEnsemble(
            pdb_id="TST",
            coords=np.random.default_rng(0).normal(size=(5, 6, 3)).astype(np.float32),
            elements=np.asarray(["C", "H", "N", "O", "S"], dtype="U2"),
            atom_names=np.asarray(["CA", "HA", "N", "O", "SG"], dtype="U4"),
            residue_index=np.zeros(5, dtype=np.int32),
            residue_names=np.asarray(["ALA"] * 5, dtype="U3"),
            energy_total=np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
            energy_terms={"bond": np.arange(5.0)},
            source=np.asarray(["torsion"] * 5, dtype="U16"),
            reference_energy=0.0,
            sequence="A",
        )
        self.assertAlmostEqual(float(ens.relative_energy.max()), 40.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = ens.save(os.path.join(tmp, "t.npz"))
            back = ProteinEnsemble.load(path)
        np.testing.assert_allclose(back.coords, ens.coords, atol=1e-5)
        np.testing.assert_allclose(back.energy_total, ens.energy_total)
        self.assertEqual(list(back.elements), list(ens.elements))
        self.assertIn("bond", back.energy_terms)


if __name__ == "__main__":
    unittest.main()
