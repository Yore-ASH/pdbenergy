"""Measure train/test overlap directly, without training a model.

Why this exists
---------------
The usual way to "show" that a frame-level split leaks is to train twice and
compare test errors.  That is expensive, and it is *indirect*: the size of the
gap depends on how much capacity the model has to exploit the leakage.  A small
linear model on global descriptors barely benefits, so the gap can look
reassuringly small even when the split is badly wrong.

This module measures the thing itself: **how close is each test conformation to
the nearest training conformation, and does that neighbour happen to be a
different frame of the same protein?**

Two numbers come out, and both are unambiguous:

``same_protein_fraction``
    Fraction of test frames whose nearest training neighbour belongs to the
    *same protein*.  Under an honest protein-level split this is exactly 0.0 by
    construction; under a frame-level split it is close to 1.0.  A test point
    whose nearest neighbour is a near-copy of itself is not a test of
    generalisation.

``median_nn_distance``
    Median distance (in standardised descriptor space) to the nearest training
    frame.  Tiny values mean the "test" set is a memory test.

The descriptor is the 24-dimensional, rotation- and permutation-invariant
:func:`pdbenergy.features.global_descriptors` vector, standardised with training
statistics.  It is cheap, needs no model, and cannot be accused of flattering a
particular architecture.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .features import global_descriptors


def descriptor_matrix(ensembles: dict, rows: Sequence[tuple[str, int]]) -> np.ndarray:
    """Featurise a list of (protein, frame) rows with the global descriptors."""
    out = np.zeros((len(rows), 24), dtype=np.float64)
    for i, (protein, frame) in enumerate(rows):
        ensemble = ensembles[protein]
        out[i] = global_descriptors(
            np.asarray(ensemble.coords[frame], dtype=np.float64),
            [str(e) for e in ensemble.elements],
        )
    return out


def nearest_neighbour_leakage(
    ensembles: dict,
    train_rows: Sequence[tuple[str, int]],
    test_rows: Sequence[tuple[str, int]],
    *,
    near_threshold: float = 0.05,
) -> dict:
    """Compare every test frame with the training set.

    Parameters
    ----------
    near_threshold:
        A standardised descriptor distance below which two conformations are
        called "near-duplicates".  0.05 is deliberately strict.
    """
    if not train_rows or not test_rows:
        return {}

    x_train = descriptor_matrix(ensembles, train_rows)
    x_test = descriptor_matrix(ensembles, test_rows)

    # Standardise with TRAINING statistics only - using all the data here would
    # be the very leakage this function is trying to detect.
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    train_proteins = np.asarray([p for p, _ in train_rows])
    test_proteins = np.asarray([p for p, _ in test_rows])

    # (n_test, n_train) distances. For a few thousand frames this is a few
    # million entries - a single BLAS-backed reduction.
    d2 = (
        np.einsum("ij,ij->i", x_test, x_test)[:, None]
        + np.einsum("ij,ij->i", x_train, x_train)[None, :]
        - 2.0 * (x_test @ x_train.T)
    )
    np.maximum(d2, 0.0, out=d2)
    dist = np.sqrt(d2)

    nn_index = np.argmin(dist, axis=1)
    nn_distance = dist[np.arange(dist.shape[0]), nn_index]
    same_protein = train_proteins[nn_index] == test_proteins

    # Also report the best *cross-protein* distance, which is the honest measure
    # of "how far is this test point from anything genuinely new".
    cross = dist.copy()
    for protein in np.unique(test_proteins):
        mask = train_proteins == protein
        if mask.any():
            cross[test_proteins == protein][:, mask] = np.inf
    cross_nn = cross.min(axis=1)

    return {
        "n_train": int(len(train_rows)),
        "n_test": int(len(test_rows)),
        "same_protein_fraction": float(np.mean(same_protein)),
        "median_nn_distance": float(np.median(nn_distance)),
        "min_nn_distance": float(np.min(nn_distance)),
        "p05_nn_distance": float(np.percentile(nn_distance, 5)),
        "near_duplicate_fraction": float(np.mean(nn_distance < near_threshold)),
        "median_cross_protein_distance": float(
            np.median(cross_nn[np.isfinite(cross_nn)])
        ) if np.isfinite(cross_nn).any() else float("nan"),
    }


def compare_split_modes(ensembles: dict, cfg, *, verbose: bool = True) -> dict:
    """Run the leakage measurement for both split modes and report the contrast."""
    from .dataset import build_bundle

    out: dict[str, dict] = {}
    for mode in ("protein", "frame"):
        cfg.train.split_mode = mode
        bundle = build_bundle(ensembles, cfg, split_mode=mode, verbose=False)
        stats = nearest_neighbour_leakage(
            ensembles, bundle.rows["train"], bundle.rows["test"]
        )
        out[mode] = stats
        if verbose and stats:
            print(
                f"  [{mode:>7}] test frames whose nearest training neighbour is the "
                f"SAME protein: {100 * stats['same_protein_fraction']:5.1f}% | "
                f"median NN distance {stats['median_nn_distance']:.3f} | "
                f"near-duplicates (<0.05) {100 * stats['near_duplicate_fraction']:5.1f}%",
                flush=True,
            )
    return out
