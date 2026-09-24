"""Evaluation: honest metrics, per-group breakdowns, and plots.

Three things are reported that a single test MAE would hide:

1. **Per-protein breakdown** - does the model work on the held-out proteins, or
   is the average dominated by one easy small protein?
2. **Per-source breakdown** - MD frames, torsional decoys and NMR models are
   very different distributions.  A model can look fine overall while failing
   completely on the high-energy decoys that matter for scoring.
3. **Prediction-vs-truth parity** - if the point cloud is compressed towards the
   mean, the model is under-fitting the energy range even when R^2 looks decent.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Sequence

import matplotlib
matplotlib.use("Agg")           # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import DataBundle, SPLIT_NAMES, collate_for, load_ensembles, build_bundle
from .train import forward_batch, load_checkpoint, regression_metrics
from .config import Config


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    bundle: DataBundle,
    split: str,
    *,
    descriptor_mode: bool,
    batch_size: int = 32,
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray]:
    """Predict every sample of one split and return truth/prediction/metadata."""
    ds = bundle.dataset(split, descriptor_mode=descriptor_mode)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_for(descriptor_mode))
    model.eval()
    preds, targets, proteins, frames, sources = [], [], [], [], []
    for batch in loader:
        out = forward_batch(model, batch, descriptor_mode)
        preds.append(out.detach().cpu().numpy())
        targets.append(batch["y"].numpy())
        proteins.extend(batch["protein"])
        frames.extend(batch["frame"])
        sources.extend(batch["source"])
    y_true = bundle.denormalise(np.concatenate(targets)) if targets else np.zeros(0)
    y_pred = bundle.denormalise(np.concatenate(preds)) if preds else np.zeros(0)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "protein": np.asarray(proteins),
        "frame": np.asarray(frames, dtype=np.int64),
        "source": np.asarray(sources),
    }


def rank_discrimination(data: dict, min_frames: int = 5) -> dict:
    """Within-protein rank correlation - "can it pick the better conformer?".

    Averaging predictions across all proteins hides the question that actually
    matters for a scoring function: given several conformations *of the same
    protein*, does the model rank them correctly?  A model can have an
    impressive global R^2 (it has learned which protein is which) while being
    useless within a single protein.
    """
    per_protein: dict[str, float] = {}
    for protein in sorted(set(data["protein"].tolist())):
        mask = data["protein"] == protein
        if int(mask.sum()) < min_frames:
            continue
        y_true = data["y_true"][mask]
        y_pred = data["y_pred"][mask]
        if np.std(y_true) < 1e-9 or np.std(y_pred) < 1e-9:
            continue
        rank_true = np.argsort(np.argsort(y_true)).astype(np.float64)
        rank_pred = np.argsort(np.argsort(y_pred)).astype(np.float64)
        per_protein[str(protein)] = float(np.corrcoef(rank_true, rank_pred)[0, 1])
    values = np.asarray(list(per_protein.values()), dtype=np.float64)
    return {
        "per_protein_spearman": per_protein,
        "mean_spearman": float(values.mean()) if values.size else float("nan"),
        "median_spearman": float(np.median(values)) if values.size else float("nan"),
        "fraction_above_0.5": float((values > 0.5).mean()) if values.size else float("nan"),
    }


def constant_baseline(bundle: DataBundle, split: str) -> dict:
    """Metrics for the dumbest possible model: always predict the training mean.

    A metric without this number is uninterpretable.  If the network's MAE is
    150 kcal/mol and the constant predictor's is also 150, the network has
    learned nothing about conformation; it has only learned the dataset mean.
    """
    y_true = bundle.y_raw[split]
    if y_true.size == 0:
        return {}
    prediction = np.full_like(y_true, bundle.norm_mean)
    metrics = regression_metrics(y_true, prediction).to_dict()
    metrics["constant_value"] = float(bundle.norm_mean)
    return metrics


def _group_metrics(labels: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        if not mask.any():
            continue
        out[str(label)] = regression_metrics(y_true[mask], y_pred[mask]).to_dict()
    return out


def evaluate_split(
    model: torch.nn.Module,
    bundle: DataBundle,
    split: str,
    *,
    descriptor_mode: bool,
    device: torch.device | str = "cpu",
    batch_size: int = 32,
) -> dict:
    """Full evaluation report for one split."""
    data = collect_predictions(
        model, bundle, split, descriptor_mode=descriptor_mode,
        batch_size=batch_size, device=device,
    )
    overall = regression_metrics(data["y_true"], data["y_pred"]).to_dict()
    report = {
        "split": split,
        "overall": overall,
        "per_protein": _group_metrics(data["protein"], data["y_true"], data["y_pred"]),
        "per_source": _group_metrics(data["source"], data["y_true"], data["y_pred"]),
        "rank_discrimination": rank_discrimination(data),
        "energy_range": {
            "true_min": float(np.min(data["y_true"])) if data["y_true"].size else None,
            "true_max": float(np.max(data["y_true"])) if data["y_true"].size else None,
            "pred_min": float(np.min(data["y_pred"])) if data["y_pred"].size else None,
            "pred_max": float(np.max(data["y_pred"])) if data["y_pred"].size else None,
        },
    }
    report["_data"] = data
    return report


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

PLOT_STYLE = {
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
}


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_parity(data: dict, path: str, title: str) -> str:
    """Predicted vs true energy, coloured by protein, with a y=x reference."""
    y_true, y_pred = data["y_true"], data["y_pred"]
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(5.4, 5.0))
        proteins = sorted(set(data["protein"].tolist()))
        cmap = plt.get_cmap("tab20")
        for i, protein in enumerate(proteins):
            mask = data["protein"] == protein
            ax.scatter(y_true[mask], y_pred[mask], s=9, alpha=0.75,
                       color=cmap(i % 20), label=protein, edgecolors="none")
        lo = float(min(y_true.min(), y_pred.min())) if y_true.size else 0.0
        hi = float(max(y_true.max(), y_pred.max())) if y_true.size else 1.0
        pad = 0.05 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.0, label="y = x")
        ax.set_xlabel("true conformational energy (kcal/mol)")
        ax.set_ylabel("predicted (kcal/mol)")
        metrics = regression_metrics(y_true, y_pred)
        ax.set_title(
            f"{title}\nMAE {metrics.mae:.2f} | RMSE {metrics.rmse:.2f} | "
            f"R2 {metrics.r2:.3f} | rho {metrics.spearman:.3f}",
            fontsize=9,
        )
        ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper left")
        return _save(fig, path)


def plot_residuals(data: dict, path: str, title: str) -> str:
    """Error distribution: is the model biased, and how heavy are the tails?"""
    err = data["y_pred"] - data["y_true"]
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
        axes[0].hist(err, bins=40, color="#4477aa", alpha=0.85)
        axes[0].axvline(0.0, color="k", lw=1.0, ls="--")
        axes[0].set_xlabel("prediction error (kcal/mol)")
        axes[0].set_ylabel("count")
        axes[0].set_title(f"{title}: error distribution", fontsize=9)

        axes[1].scatter(data["y_true"], err, s=9, alpha=0.6, color="#cc6677", edgecolors="none")
        axes[1].axhline(0.0, color="k", lw=1.0, ls="--")
        axes[1].set_xlabel("true energy (kcal/mol)")
        axes[1].set_ylabel("error (kcal/mol)")
        axes[1].set_title("error vs energy (heteroscedasticity check)", fontsize=9)
        return _save(fig, path)


def plot_learning_curve(history: dict, path: str, selected_epoch: int | None = None) -> str:
    """Loss curves on a log scale plus validation MAE in kcal/mol.

    The vertical line marks the epoch whose weights were **kept** (the one with
    the lowest validation loss, which is what early stopping restores).  That is
    not necessarily the epoch with the lowest validation MAE, and confusing the
    two flatters the model.
    """
    epochs = history.get("epoch", [])
    if not epochs:
        return path
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
        axes[0].plot(epochs, history["train_loss"], label="train")
        axes[0].plot(epochs, history["val_loss"], label="val")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss (standardised target)")
        axes[0].set_title("learning curve (log scale)", fontsize=9)
        axes[0].legend(fontsize=8)

        axes[1].plot(epochs, history["val_mae"], color="#228833")
        if selected_epoch:
            keep = min(max(1, int(selected_epoch)), len(epochs)) - 1
            axes[1].axvline(epochs[keep], color="k", ls=":", lw=1.2,
                            label=f"kept epoch {epochs[keep]}")
            axes[1].legend(fontsize=8)
        best_mae = min(history["val_mae"]) if history["val_mae"] else float("nan")
        axes[1].set_title(
            f"validation MAE (min over epochs {best_mae:.2f} kcal/mol)", fontsize=9)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("MAE (kcal/mol)")

        axes[2].plot(epochs, history["lr"], color="#aa3377")
        axes[2].set_yscale("log")
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("learning rate")
        axes[2].set_title("learning-rate schedule", fontsize=9)
        return _save(fig, path)


def plot_group_bars(report: dict, path: str, key: str, title: str, value: str = "mae") -> str:
    """Horizontal bar chart of a metric across proteins or sources."""
    groups = report.get(key, {})
    if not groups:
        return path
    labels = list(groups.keys())
    values = [groups[k].get(value, float("nan")) for k in labels]
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(5.6, max(2.4, 0.30 * len(labels) + 1.2)))
        ax.barh(labels, values, color="#4477aa", alpha=0.9)
        for i, v in enumerate(values):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=7)
        ax.set_xlabel(f"{value} (kcal/mol)")
        ax.set_title(title, fontsize=9)
        return _save(fig, path)


def plot_energy_distribution(bundle: DataBundle, path: str) -> str:
    """Where the training data actually lives - always look at this first."""
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
        all_rows = [r for split in SPLIT_NAMES for r in bundle.rows[split]]
        y = np.asarray([bundle.raw_target(p, f) for p, f in all_rows])
        axes[0].hist(y, bins=60, color="#4477aa", alpha=0.9)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("relative conformational energy (kcal/mol)")
        axes[0].set_ylabel("count (log)")
        axes[0].set_title("target distribution (all splits)", fontsize=9)

        sources = np.asarray([str(bundle.ensembles[p].source[f]) for p, f in all_rows])
        order = sorted(set(sources.tolist()))
        data = [y[sources == s] for s in order]
        try:
            # matplotlib >= 3.9 renamed `labels` -> `tick_labels`, and >= 3.10
            # deprecated `vert` in favour of `orientation`.
            axes[1].boxplot(data, tick_labels=order, orientation="horizontal",
                            showfliers=False)
        except TypeError:  # older matplotlib
            axes[1].boxplot(data, labels=order, vert=False, showfliers=False)
        axes[1].set_xlabel("relative energy (kcal/mol)")
        axes[1].set_title("per sampling source", fontsize=9)
        return _save(fig, path)


def plot_per_protein_range(bundle: DataBundle, path: str) -> str:
    """Energy span per protein - shows how much of the range each protein covers."""
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.4, max(2.4, 0.28 * len(bundle.ensembles) + 1.2)))
        names, lows, highs = [], [], []
        for p, ens in sorted(bundle.ensembles.items()):
            de = ens.relative_energy
            names.append(p)
            lows.append(float(np.percentile(de, 5)))
            highs.append(float(np.percentile(de, 95)))
        ax.barh(names, np.asarray(highs) - np.asarray(lows), left=lows, color="#228833", alpha=0.85)
        ax.set_xlabel("relative energy, 5th-95th percentile (kcal/mol)")
        ax.set_title("conformational energy span per protein", fontsize=9)
        return _save(fig, path)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def evaluate_run(
    run_dir: str,
    *,
    interim_dir: str = "data/interim",
    out_dir: str | None = None,
    splits: Sequence[str] = ("val", "test"),
    verbose: bool = True,
) -> dict:
    """Evaluate a saved checkpoint and write metrics + figures."""
    out_dir = out_dir or os.path.join(run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    payload = load_checkpoint(os.path.join(run_dir, "checkpoint.pt"), device="cpu")
    model = payload["model"]
    descriptor_mode = bool(payload.get("descriptor_mode", False))

    cfg_path = os.path.join(run_dir, "config.json")
    cfg = Config.load(cfg_path) if os.path.exists(cfg_path) else Config()
    # The checkpoint is authoritative about the input layout: if the feature
    # config changed after training, rebuilding graphs from config.json would
    # feed the model the wrong number of edge channels.
    if payload.get("_feature_config") is not None:
        cfg.features = payload["_feature_config"]

    ensembles = load_ensembles(interim_dir, verbose=False)
    # Rebuild the splits exactly as training saw them: use the stored protein
    # split so evaluation cannot silently disagree with the training run.
    bundle = build_bundle(
        ensembles, cfg,
        split_mode=payload.get("split_mode", cfg.train.split_mode),
        verbose=False,
    )
    if payload.get("protein_splits"):
        stored = payload["protein_splits"]
        if all(stored.get(k) for k in SPLIT_NAMES):
            bundle = _rebuild_with_proteins(ensembles, cfg, stored, payload.get("split_mode", "protein"))

    bundle.norm_mean = float(payload["norm_mean"])
    bundle.norm_std = float(payload["norm_std"])

    report: dict = {
        "run_dir": run_dir,
        "model_kind": payload.get("model_kind"),
        "descriptor_mode": descriptor_mode,
        "target": payload.get("target"),
        "split_mode": payload.get("split_mode"),
        "n_parameters": payload.get("n_parameters"),
        "best_epoch": payload.get("best_epoch"),
        "protein_splits": payload.get("protein_splits"),
        "normalisation": {"mean": bundle.norm_mean, "std": bundle.norm_std},
        "splits": {},
        "baselines": {
            split: constant_baseline(bundle, split) for split in SPLIT_NAMES
            if bundle.rows.get(split)
        },
        "artifacts": {},
    }

    for split in splits:
        if not bundle.rows.get(split):
            continue
        result = evaluate_split(model, bundle, split, descriptor_mode=descriptor_mode)
        data = result.pop("_data")
        report["splits"][split] = result
        if verbose:
            m = result["overall"]
            base = report["baselines"].get(split, {})
            print(
                f"  [{split}] n={m['n']} MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} "
                f"R2={m['r2']:.4f} rho={m['spearman']:.4f}"
                + (f" | constant-baseline MAE={base['mae']:.3f}" if base else ""),
                flush=True,
            )
        report["artifacts"][f"parity_{split}"] = plot_parity(
            data, os.path.join(out_dir, f"parity_{split}.png"),
            f"{payload.get('model_kind')} on {split} ({payload.get('split_mode')} split)",
        )
        report["artifacts"][f"residuals_{split}"] = plot_residuals(
            data, os.path.join(out_dir, f"residuals_{split}.png"), split
        )
        report["artifacts"][f"per_protein_{split}"] = plot_group_bars(
            result, os.path.join(out_dir, f"per_protein_{split}.png"),
            "per_protein", f"MAE per protein ({split})",
        )
        report["artifacts"][f"per_source_{split}"] = plot_group_bars(
            result, os.path.join(out_dir, f"per_source_{split}.png"),
            "per_source", f"MAE per sampling source ({split})",
        )

    if payload.get("history"):
        report["artifacts"]["learning_curve"] = plot_learning_curve(
            payload["history"], os.path.join(out_dir, "learning_curve.png"),
            selected_epoch=payload.get("best_epoch"),
        )
    report["artifacts"]["energy_distribution"] = plot_energy_distribution(
        bundle, os.path.join(out_dir, "energy_distribution.png")
    )
    report["artifacts"]["per_protein_range"] = plot_per_protein_range(
        bundle, os.path.join(out_dir, "per_protein_range.png")
    )

    metrics_path = os.path.join(out_dir, "metrics.json")
    # Via jsonutil: several metrics are NaN when undefined (R^2 with zero target
    # variance, a correlation over one sample).  Python would write the bare token
    # NaN, which is not valid JSON and breaks every strict reader - the browser
    # GUI failed with "Unexpected token 'N' ... is not valid JSON" because of it.
    from .jsonutil import dump_json

    dump_json(metrics_path, report)
    report["artifacts"]["metrics"] = metrics_path
    return report


def _rebuild_with_proteins(
    ensembles: dict, cfg: Config, protein_splits: dict[str, list[str]], split_mode: str
) -> DataBundle:
    """Rebuild a bundle from an explicit protein split (used at evaluation time)."""
    from .dataset import DataBundle as _Bundle, SPLIT_NAMES as _SPLITS

    bundle = _Bundle(
        ensembles=ensembles, cfg=cfg, split_mode=split_mode,
        protein_splits={k: list(protein_splits.get(k, [])) for k in _SPLITS},
        leak_free_proteins={k: list(protein_splits.get(k, [])) for k in _SPLITS},
    )
    rows: dict[str, list[tuple[str, int]]] = {k: [] for k in _SPLITS}
    if split_mode == "protein":
        for split in _SPLITS:
            for p in protein_splits.get(split, []):
                if p in ensembles:
                    rows[split].extend((p, f) for f in range(len(ensembles[p])))
    else:
        # For the leaky frame split, reproduce the split with the stored seed.
        proteins = sorted(ensembles)
        all_rows = [(p, f) for p in proteins for f in range(len(ensembles[p]))]
        rng = np.random.default_rng(cfg.train.seed)
        idx = rng.permutation(len(all_rows))
        n_test = max(1, int(round(len(all_rows) * cfg.train.test_fraction)))
        n_val = max(1, int(round(len(all_rows) * cfg.train.val_fraction)))
        rows["test"] = [all_rows[i] for i in idx[:n_test]]
        rows["val"] = [all_rows[i] for i in idx[n_test:n_test + n_val]]
        rows["train"] = [all_rows[i] for i in idx[n_test + n_val:]]
    bundle.rows = rows
    bundle.y_raw = {
        split: np.asarray([bundle.raw_target(p, f) for p, f in rows[split]], dtype=np.float64)
        for split in _SPLITS
    }
    train_y = bundle.y_raw["train"]
    bundle.norm_mean = float(np.mean(train_y)) if train_y.size else 0.0
    bundle.norm_std = float(np.std(train_y)) if train_y.size else 1.0
    if bundle.norm_std < 1e-8:
        bundle.norm_std = 1.0
    return bundle
