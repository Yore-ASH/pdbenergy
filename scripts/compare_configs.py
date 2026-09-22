# -*- coding: utf-8 -*-
"""Ablate config settings against one pinned split, at an equal training budget.

The point is to make "what should I change?" answerable without a 70-hour
labelling run.  Every variant sees the *same* frozen protein split, the same
seed, and the same epoch budget, so differences are attributable to the setting
and not to the data or the schedule.

    python scripts/compare_configs.py --epochs 8 \
        --variant baseline \
        --variant rbf64=features.n_rbf=64 \
        --variant rbf64_local=features.n_rbf=64,features.rbf_end=3.0

Each ``--variant`` is ``name=k=v[,k=v...]`` where ``k`` is a dotted path into the
config (``features.n_rbf``, ``model.hidden_dim``, ``train.learning_rate`` ...).
Values are coerced to the type of the existing dataclass field.

A caveat worth stating plainly: at an 8-epoch budget these models are nowhere
near converged, so this measures *relative* promise at equal cost, not final
accuracy.  A variant that wins here is worth a proper run; it is not a result.

Notes
-----
* Changing ``features.*`` changes the graph layout, so the cached graphs are
  invalid and a fresh cache is built per variant. ``--cache-dir`` keeps those
  throwaway caches out of ``data/processed``.
* Val MAE is used for the verdict (test is reported but never used to choose).
"""

from __future__ import annotations

import argparse
import dataclasses
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
from pdbenergy.dataset import build_bundle, load_ensembles  # noqa: E402
from pdbenergy.evaluate import constant_baseline, evaluate_split  # noqa: E402
from pdbenergy.iterate import apply_split                 # noqa: E402
from pdbenergy.train import train_model                   # noqa: E402


def parse_variant(spec: str) -> tuple[str, dict[str, str]]:
    name, _, rest = spec.partition("=")
    overrides: dict[str, str] = {}
    for item in rest.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition("=")
        if not _:
            raise SystemExit(f"variant {name!r}: expected key=value, got {item!r}")
        overrides[key.strip()] = value.strip()
    return name.strip() or "variant", overrides


def coerce(current, value: str):
    """Cast ``value`` to the type of the existing field, so dataclasses stay valid."""
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(float(value))
    if isinstance(current, float):
        return float(value)
    if isinstance(current, (tuple, list)):
        return type(current)(float(v) for v in value.split("|"))
    return value


def set_dotted(cfg: Config, dotted: str, value: str) -> tuple[str, object, object]:
    parts = dotted.split(".")
    if len(parts) != 2:
        raise SystemExit(f"expected section.field, got {dotted!r}")
    section, field = parts
    target = getattr(cfg, section, None)
    if target is None or not dataclasses.is_dataclass(target):
        raise SystemExit(f"no config section {section!r}")
    if not hasattr(target, field):
        raise SystemExit(f"no field {field!r} in {section!r}")
    old = getattr(target, field)
    new = coerce(old, value)
    setattr(target, field, new)
    return dotted, old, new


def describe(cfg: Config) -> str:
    return (f"hidden={cfg.model.hidden_dim} int={cfg.model.n_interactions} "
            f"n_rbf={cfg.features.n_rbf} cutoff={cfg.features.cutoff} "
            f"rbf_end={cfg.features.rbf_end} max_nbr={cfg.features.max_neighbors}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="outputs/schnet_protein/checkpoint.pt",
                        help="source of the frozen split and the base feature config")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--variant", action="append", default=None,
                        help="name=k=v[,k=v...]   (repeatable)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cache-dir", default="logs/ablation_cache",
                        help="throwaway graph caches (features.* changes invalidate them)")
    parser.add_argument("--out", default="outputs/config_ablation.json")
    parser.add_argument("--out-dir", default="logs/config_ablation")
    parser.add_argument("--keep-runs", action="store_true",
                        help="keep each variant's checkpoints (they are ~0.7 MB each)")
    args = parser.parse_args()

    specs = args.variant or ["baseline"]
    if not os.path.exists(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint!r}")
        return 1

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    frozen = payload["protein_splits"]
    ensembles = load_ensembles(args.interim_dir, verbose=False)

    # Baseline config: the checkpoint's own feature layout, so 'baseline' here
    # reproduces the shipped model's input pipeline.
    base_features = FeatureConfig(**payload["feature_config"])
    base_model = payload["model_config"]

    baseline_mae = None
    results = []
    print(f"frozen split: {len(frozen['train'])}/{len(frozen['val'])}/{len(frozen['test'])} "
          f"proteins   budget: {args.epochs} epochs per variant\n")
    header = f"{'variant':<26} {'val MAE':>9} {'test MAE':>9} {'train loss':>11} {'sec':>6}  config"
    print(header)
    print("-" * len(header))

    for spec in specs:
        name, overrides = parse_variant(spec)

        cfg = Config()
        cfg.features = FeatureConfig(**dataclasses.asdict(base_features))
        cfg.model.hidden_dim = base_model.get("hidden_dim", cfg.model.hidden_dim)
        cfg.model.n_interactions = base_model.get("n_interactions", cfg.model.n_interactions)
        cfg.model.cutoff = base_model.get("cutoff", cfg.model.cutoff)
        cfg.train.epochs = args.epochs
        cfg.train.seed = args.seed
        cfg.train.log_every = max(1, args.epochs)
        cfg.train.graph_cache_dir = args.cache_dir

        applied = []
        for dotted, value in overrides.items():
            field, old, new = set_dotted(cfg, dotted, value)
            applied.append(f"{field}: {old} -> {new}")

        bundle = build_bundle(ensembles, cfg, verbose=False)
        apply_split(bundle, frozen)

        t0 = time.time()
        result = train_model(bundle, out_dir=os.path.join(args.out_dir, name), verbose=False)
        wall = time.time() - t0

        descriptor_mode = cfg.model.kind == "mlp"
        val = evaluate_split(result.model, bundle, "val", descriptor_mode=descriptor_mode)
        test = evaluate_split(result.model, bundle, "test", descriptor_mode=descriptor_mode)
        train_loss = float(min(result.history["train_loss"])) if result.history["train_loss"] else float("nan")
        val_mae = float(val["overall"]["mae"])
        test_mae = float(test["overall"]["mae"])

        if name == "baseline":
            baseline_mae = val_mae

        results.append({
            "variant": name,
            "overrides": applied,
            "val_mae": val_mae,
            "test_mae": test_mae,
            "test_r2": float(test["overall"]["r2"]),
            "test_spearman": float(test["overall"]["spearman"]),
            "within_protein_rho": float(test["rank_discrimination"]["mean_spearman"]),
            "train_loss": train_loss,
            "prediction_span_ratio": _span_ratio(test),
            "epochs": args.epochs,
            "wall_time_s": wall,
            "config": describe(cfg),
            "constant_baseline_mae": float(constant_baseline(bundle, "val").get("mae", float("nan"))),
        })
        print(f"{name:<26} {val_mae:>9.2f} {test_mae:>9.2f} {train_loss:>11.4f} "
              f"{wall:>6.0f}  {describe(cfg)}")

    # ---- verdict ---------------------------------------------------------- #
    print()
    baseline = next((r for r in results if r["variant"] == "baseline"), None)
    if baseline and len(results) > 1:
        better = [r for r in results
                  if r["variant"] != "baseline" and r["val_mae"] < baseline["val_mae"]]
        if better:
            best = min(better, key=lambda r: r["val_mae"])
            gain = baseline["val_mae"] - best["val_mae"]
            print(f"best non-baseline variant: {best['variant']} "
                  f"({gain:+.2f} kcal/mol val MAE vs baseline, "
                  f"{100 * gain / baseline['val_mae']:.0f}% better)")
            print("  -> a CONFIG change beats adding data at equal budget. "
                  "Fix this before paying for more labelling.")
        else:
            print("no config change beat the baseline at this budget.")
            print("  -> the settings tried here are not the bottleneck, or 8 epochs is too "
                  "short to separate them. Try more epochs before concluding.")
    print(f"  constant predictor val MAE = "
          f"{results[0]['constant_baseline_mae']:.2f} kcal/mol (the bar to beat)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")
    if not args.keep_runs:
        print(f"(runs left in {args.out_dir}; delete when done)")
    return 0


def _span_ratio(report: dict) -> float:
    rng = report.get("energy_range", {})
    true_span = (rng.get("true_max") or 0) - (rng.get("true_min") or 0)
    pred_span = (rng.get("pred_max") or 0) - (rng.get("pred_min") or 0)
    return float(pred_span / true_span) if true_span else float("nan")


if __name__ == "__main__":
    sys.exit(main())
