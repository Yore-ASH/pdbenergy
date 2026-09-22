"""Training loop: reproducible, checkpointed, early-stopped.

The whole loop is deliberately explicit rather than a framework call, because
every line here corresponds to a decision that matters and is discussed in
``TeachFlow.md``:

* seeding every RNG (Python, NumPy, Torch) so a run can be reproduced;
* standardising the target so the loss is scale-free;
* a Huber loss so a handful of very high-energy distorted structures do not
  dominate the gradient;
* AdamW with weight decay (decoupled, unlike classic Adam+L2);
* a learning-rate scheduler driven by validation loss;
* early stopping that restores the *best* weights, not the last ones;
* gradient clipping because message passing can produce large gradients;
* checkpointing the config, the normalisation statistics and the split lists
  *with* the weights, so a checkpoint is self-describing.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .dataset import DataBundle, collate_for
from .models import build_model, count_parameters, edge_dims_for


# --------------------------------------------------------------------------- #
# Reproducibility and devices
# --------------------------------------------------------------------------- #


def set_seed(seed: int) -> None:
    """Seed every random number generator we use.

    Note the honest caveat: on CUDA, ``use_deterministic_algorithms`` is needed
    for *bit-exact* reproducibility, and some cuDNN kernels remain
    non-deterministic.  CPU training is deterministic here.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    """Turn a config string into a concrete ``torch.device``."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


# --------------------------------------------------------------------------- #
# Batch plumbing
# --------------------------------------------------------------------------- #


def forward_batch(model: nn.Module, batch: dict, descriptor_mode: bool) -> torch.Tensor:
    """Call either model family with the right arguments."""
    if descriptor_mode:
        return model(batch["x"])
    return model(
        z=batch["z"],
        edge_index=batch["edge_index"],
        edge_attr=batch["edge_attr"],
        batch=batch["batch"],
        n_graphs=batch["n_graphs"],
    )


def build_loss(cfg: Config) -> nn.Module:
    """Huber (smooth L1) by default: quadratic near zero, linear in the tails.

    With a squared loss, one badly clashing decoy at +2000 kcal/mol produces a
    gradient thousands of times larger than a typical sample and drags the whole
    fit.  Huber caps the influence of those outliers while keeping a smooth,
    well-behaved gradient near the optimum.
    """
    if cfg.train.loss == "mse":
        return nn.MSELoss()
    if cfg.train.loss == "huber":
        return nn.HuberLoss(delta=cfg.train.huber_delta)
    raise ValueError(f"unknown loss {cfg.train.loss!r}")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


@dataclass
class Metrics:
    """Regression metrics, all in kcal/mol except the dimensionless scores."""

    n: int = 0
    mae: float = float("nan")
    rmse: float = float("nan")
    r2: float = float("nan")
    pearson: float = float("nan")
    spearman: float = float("nan")
    bias: float = float("nan")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """MAE / RMSE / R^2 / Pearson / Spearman / bias.

    Spearman's rank correlation is included because for many applications
    (ranking decoys, scoring candidate structures) only the *order* matters.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    n = y_true.size
    if n == 0:
        return Metrics()
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    var = float(np.var(y_true))
    r2 = float(1.0 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)) if var > 0 else float("nan")
    if n > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        # Spearman = Pearson on ranks; avoids a SciPy dependency.
        rank_true = np.argsort(np.argsort(y_true)).astype(np.float64)
        rank_pred = np.argsort(np.argsort(y_pred)).astype(np.float64)
        spearman = float(np.corrcoef(rank_true, rank_pred)[0, 1])
    else:
        pearson = spearman = float("nan")
    return Metrics(n=n, mae=mae, rmse=rmse, r2=r2, pearson=pearson, spearman=spearman,
                   bias=float(np.mean(err)))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass
class TrainResult:
    model: nn.Module
    history: dict[str, list[float]] = field(default_factory=dict)
    best_epoch: int = 0
    best_val_mae: float = float("nan")
    n_parameters: int = 0
    device: str = "cpu"
    wall_time_s: float = 0.0


def evaluate_loader(model: nn.Module, loader: DataLoader, device, descriptor_mode: bool) -> tuple[float, np.ndarray, np.ndarray]:
    """Run the model over a loader; returns (mean loss, y_true, y_pred) in kcal/mol."""
    model.eval()
    loss_fn = nn.MSELoss()  # reporting metric only; training uses its own loss
    total, count = 0.0, 0
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            y = batch["y"].to(device)
            out = forward_batch(model, batch, descriptor_mode)
            total += float(loss_fn(out, y)) * y.numel()
            count += y.numel()
            preds.append(out.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
    if count == 0:
        return float("nan"), np.zeros(0), np.zeros(0)
    return total / count, np.concatenate(targets), np.concatenate(preds)


def train_model(
    bundle: DataBundle,
    *,
    descriptor_mode: bool = False,
    out_dir: str | None = None,
    verbose: bool = True,
) -> TrainResult:
    """Train one model and return it together with its learning curve."""
    cfg = bundle.cfg
    tcfg = cfg.train
    set_seed(tcfg.seed)
    device = resolve_device(tcfg.device)

    # Build (or load) the graph cache once for every sample in the bundle.
    cache = None
    if not descriptor_mode and tcfg.cache_graphs:
        from .graphcache import load_or_build

        cache = load_or_build(
            bundle.ensembles, bundle.all_rows(), cfg.features,
            tcfg.graph_cache_dir, enabled=True, verbose=verbose,
        )

    train_ds = bundle.dataset("train", descriptor_mode=descriptor_mode, cache=cache)
    val_ds = bundle.dataset("val", descriptor_mode=descriptor_mode, cache=cache)
    collate = collate_for(descriptor_mode)

    generator = torch.Generator()
    generator.manual_seed(tcfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=tcfg.batch_size, shuffle=True, collate_fn=collate,
        num_workers=tcfg.num_workers, generator=generator, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg.batch_size, shuffle=False, collate_fn=collate,
        num_workers=tcfg.num_workers,
    )

    model = build_model(cfg.model, **edge_dims_for(cfg.features)).to(device)
    n_params = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay
    )
    if tcfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(5, tcfg.patience // 4)
        )
    loss_fn = build_loss(cfg)

    history: dict[str, list[float]] = {
        "epoch": [], "train_loss": [], "val_loss": [], "val_mae": [], "lr": []
    }
    best_val = float("inf")
    best_val_mae = float("nan")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    epochs_without_improvement = 0
    start = time.time()

    if verbose:
        print(
            f"  model={cfg.model.kind} params={n_params:,} device={device} "
            f"train={len(train_ds)} val={len(val_ds)} batch={tcfg.batch_size} "
            f"loss={tcfg.loss}\n"
            f"  {'epoch':>5} {'train':>10} {'val':>10} {'val MAE':>10} {'lr':>10} {'sec':>6}",
            flush=True,
        )

    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        t0 = time.time()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            y = batch["y"].to(device)
            out = forward_batch(model, batch, descriptor_mode)
            loss = loss_fn(out, y)
            loss.backward()
            if tcfg.grad_clip and tcfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            optimizer.step()
            running += float(loss.detach()) * y.numel()
            seen += y.numel()
        train_loss = running / max(1, seen)

        val_loss, y_true_n, y_pred_n = evaluate_loader(model, val_loader, device, descriptor_mode)
        # Denormalise before reporting so the number is in kcal/mol.
        y_true = bundle.denormalise(y_true_n)
        y_pred = bundle.denormalise(y_pred_n)
        val_mae = regression_metrics(y_true, y_pred).mae

        if tcfg.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(float(val_mae))
        history["lr"].append(float(lr))

        improved = val_loss < best_val - tcfg.min_delta
        if improved:
            best_val = val_loss
            best_val_mae = float(val_mae)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch % tcfg.log_every == 0 or epoch == 1 or improved):
            print(
                f"  {epoch:>5} {train_loss:>10.5f} {val_loss:>10.5f} {val_mae:>10.3f} "
                f"{lr:>10.2e} {time.time()-t0:>6.1f}",
                flush=True,
            )

        if epochs_without_improvement >= tcfg.patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break

    # Restore the best weights, never the last ones.
    model.load_state_dict(best_state)
    result = TrainResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_mae=best_val_mae,
        n_parameters=n_params,
        device=str(device),
        wall_time_s=time.time() - start,
    )

    if out_dir:
        save_checkpoint(out_dir, bundle, result, descriptor_mode=descriptor_mode)
    return result


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #


def save_checkpoint(out_dir: str, bundle: DataBundle, result: TrainResult, *, descriptor_mode: bool) -> str:
    """Write ``checkpoint.pt`` plus human-readable JSON beside it."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "checkpoint.pt")
    torch.save(
        {
            "model_state_dict": result.model.state_dict(),
            "model_config": asdict(bundle.cfg.model),
            "feature_config": asdict(bundle.cfg.features),
            "model_kind": bundle.cfg.model.kind,
            "descriptor_mode": descriptor_mode,
            "target": bundle.target,
            "norm_mean": bundle.norm_mean,
            "norm_std": bundle.norm_std,
            "protein_splits": bundle.protein_splits,
            "split_mode": bundle.split_mode,
            "history": result.history,
            "best_epoch": result.best_epoch,
            "n_parameters": result.n_parameters,
            "version": 1,
        },
        path,
    )
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(result.history, fh, indent=2)
    with open(os.path.join(out_dir, "data_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(bundle.summary(), fh, indent=2)
    bundle.cfg.save(os.path.join(out_dir, "config.json"))
    return path


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Load a checkpoint and rebuild the model it describes.

    Returns a dict with ``model``, ``payload`` and the normalisation statistics,
    so a checkpoint alone is enough to make a prediction.
    """
    from .config import FeatureConfig, ModelConfig

    payload = torch.load(path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**payload["model_config"])
    feature_cfg = FeatureConfig(**payload["feature_config"])
    model = build_model(model_cfg, **edge_dims_for(feature_cfg))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    payload["model"] = model
    payload["_model_config"] = model_cfg
    payload["_feature_config"] = feature_cfg
    return payload
