# -*- coding: utf-8 -*-
"""Measure whether more data would actually help: a learning curve.

The question
------------
"Should I spend 20-70 hours labelling 800-1200 more proteins?"  The answer is
*measurable on the data you already have*: train the same model on 25%, 50% and
100% of the training conformers and look at how validation error moves.

How to read the result
----------------------
* **Validation error falls roughly log-linearly with data size** -> the model is
  data-limited.  More labelling buys accuracy, and the slope tells you how much.
* **Validation error is flat while training error is already low** -> the model is
  *not* data-limited; it is limited by the representation, the target, or the
  optimisation.  More labelling would be wasted money until that is fixed.

This is deliberately built on the *existing* labelled data, so it costs minutes
instead of the hours a real labelling run would.

Val and test proteins are pinned to the shipped run's split (read from its
checkpoint), so the numbers here are directly comparable across fractions.

    python scripts/learning_curve.py --fractions 0.25 0.5 1.0 --epochs 8
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pdbenergy.config import Config, FeatureConfig        # noqa: E402
from pdbenergy.dataset import SPLIT_NAMES, build_bundle, load_ensembles  # noqa: E402
from pdbenergy.evaluate import constant_baseline, evaluate_split          # noqa: E402
from pdbenergy.train import train_model                    # noqa: E402


def subsample_train(rows, fraction: float, seed: int) -> list[tuple[str, int]]:
    """Take ``fraction`` of the training rows, keeping every protein represented.

    Sampling per protein (rather than globally) matters: dropping a whole fold
    would confound "less data" with "fewer folds to learn from".
    """
    if fraction >= 1.0:
        return list(rows)
    by_protein: dict[str, list[tuple[str, int]]] = {}
    for row in rows:
        by_protein.setdefault(row[0], []).append(row)
    rng = np.random.default_rng(seed)
    out: list[tuple[str, int]] = []
    for protein, group in sorted(by_protein.items()):
        n = max(1, int(round(len(group) * fraction)))
        idx = rng.permutation(len(group))[:n]
        out.extend(group[i] for i in sorted(idx))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="outputs/schnet_protein/checkpoint.pt",
                        help="its protein split and feature config pin the comparison")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="outputs/learning_curve.json")
    parser.add_argument("--out-dir", default="logs/learning_curve")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint!r} - train a model first")
        return 1

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config()
    cfg.features = FeatureConfig(**payload["feature_config"])
    cfg.model.hidden_dim = payload["model_config"].get("hidden_dim", cfg.model.hidden_dim)
    cfg.model.n_interactions = payload["model_config"].get(
        "n_interactions", cfg.model.n_interactions)
    cfg.train.epochs = args.epochs
    cfg.train.seed = args.seed
    cfg.train.log_every = max(1, args.epochs)

    ensembles = load_ensembles(args.interim_dir, verbose=False)
    bundle = build_bundle(ensembles, cfg, verbose=False)
    frozen = payload["protein_splits"]

    from pdbenergy.iterate import apply_split

    apply_split(bundle, frozen)
    full_train = bundle.rows["train"]
    print(f"frozen split: train={len(full_train)} val={len(bundle.rows['val'])} "
          f"test={len(bundle.rows['test'])}  "
          f"(proteins {len(frozen['train'])}/{len(frozen['val'])}/{len(frozen['test'])})")
    print(f"model: hidden={cfg.model.hidden_dim} interactions={cfg.model.n_interactions} "
          f"n_rbf={cfg.features.n_rbf} cutoff={cfg.features.cutoff}")
    print(f"budget: {args.epochs} epochs per point, "
          f"{len(args.fractions)} points\n")

    # The constant predictor's error on val is the bar every point must beat.
    baseline = constant_baseline(bundle, "val").get("mae", float("nan"))
    print(f"{'fraction':>9} {'train N':>8} {'train loss':>11} {'val loss':>9} "
          f"{'val MAE':>9} {'vs const':>9} {'sec':>6}")
    print("-" * 68)

    results = []
    for fraction in args.fractions:
        # Rebuild the bundle's train rows, then re-derive y and normalisation.
        rows = subsample_train(full_train, fraction, args.seed)
        frozen_sub = {k: list(v) for k, v in frozen.items()}
        bundle.rows["train"] = list(rows)
        bundle.y_raw["train"] = np.asarray(
            [bundle.raw_target(p, f) for p, f in rows], dtype=np.float64
        )
        train_y = bundle.y_raw["train"]
        bundle.norm_mean = float(train_y.mean()) if train_y.size else 0.0
        bundle.norm_std = float(train_y.std()) if train_y.size else 1.0
        if bundle.norm_std < 1e-8:
            bundle.norm_std = 1.0

        t0 = time.time()
        result = train_model(
            bundle, out_dir=os.path.join(args.out_dir, f"frac_{fraction:g}"),
            verbose=False,
        )
        wall = time.time() - t0

        val = evaluate_split(result.model, bundle, "val", descriptor_mode=False)
        train_loss = float(min(result.history["train_loss"])) if result.history["train_loss"] else float("nan")
        val_loss = float(min(result.history["val_loss"])) if result.history["val_loss"] else float("nan")
        val_mae = float(val["overall"]["mae"])

        results.append({
            "fraction": fraction,
            "n_train": len(rows),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "constant_baseline_mae": float(baseline),
            "epochs": args.epochs,
            "wall_time_s": wall,
        })
        delta = val_mae - baseline
        print(f"{fraction:>9.2f} {len(rows):>8} {train_loss:>11.4f} {val_loss:>9.4f} "
              f"{val_mae:>9.2f} {delta:>+9.2f} {wall:>6.0f}")

    # ---- verdict ---------------------------------------------------------- #
    print()
    if len(results) >= 2:
        smallest, largest = results[0], results[-1]
        drop = smallest["val_mae"] - largest["val_mae"]
        data_ratio = largest["n_train"] / max(1, smallest["n_train"])
        print(f"val MAE moves {drop:+.2f} kcal/mol while training data grows "
              f"{data_ratio:.1f}x "
              f"({smallest['n_train']} -> {largest['n_train']} conformers)")
        if drop > 0.15 * smallest["val_mae"]:
            print("  -> clearly DATA-LIMITED. More labelling should keep buying accuracy.")
            print("     Extrapolate: fit val_mae against log(N) and read off what N you need.")
        elif drop > 0.05 * smallest["val_mae"]:
            print("  -> partially data-limited. Some gain is available but it is shallow;")
            print("     run the feature-resolution experiment (TeachFlow 12.3, item 1) too.")
        else:
            print("  -> NOT data-limited. Adding conformers is unlikely to help much;")
            print("     the bottleneck is the representation / target, not the sample count.")
        print(f"\n  reference: constant predictor MAE = {baseline:.2f} kcal/mol")
        if all(r["val_mae"] >= baseline for r in results):
            print("  NOTE: every point is still worse than the constant predictor. Even a")
            print("        steep curve has to cross that line before the model is useful.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
