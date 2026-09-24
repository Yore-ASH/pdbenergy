"""Iterative training: keep adding structures, keep improving accuracy.

Why this module exists
----------------------
The single-run pipeline answers "does this work?".  This module answers the
question you actually care about once you have a pile of PDB files: *"does more
data make it better, and how do I see that?"*

Two things make a naive loop lie to you:

1. **A moving test set.** ``split_proteins`` derives the split from the list of
   proteins, so adding files re-shuffles who is in val/test.  Test MAE would then
   change for reasons that have nothing to do with the model.  This module
   **freezes the split in round 1** and appends every later-arriving protein to
   *train* only, so the test set is literally the same proteins in every round
   and the numbers are comparable.
2. **Selecting on the test set.** The "best" checkpoint is chosen by **validation**
   MAE.  Test metrics are reported, never used for selection.

Each round:

* builds ensembles for any PDB in ``data/raw`` that has no label yet
  (``--build-new``),
* rebuilds the dataset with the frozen split,
* trains, warm-starting from the previous round's checkpoint (``--warm-start``),
* evaluates on the frozen val/test split,
* appends one line to ``rounds.jsonl`` and refreshes ``rounds.md``.

Usage
-----
    # see where you are
    python -m pdbenergy.iterate --report

    # add PDB files to data/raw, then run three more rounds
    python -m pdbenergy.iterate --rounds 3 --build-new --threads 8

    # improve on an existing model without adding data (more epochs, warm start)
    python -m pdbenergy.iterate --rounds 2 --no-build-new --epochs 30

Caveat worth knowing
--------------------
Warm-starting assumes the *same* architecture.  If you decide to grow the model
(raise ``hidden_dim`` / ``cutoff``) the warm start cannot load, and the
compatibility check in :func:`pdbenergy.train.train_model` will refuse rather than
silently train half a model.  In that case start a new run directory
(``--out-dir outputs/iterative_big``) and train from scratch: more data is exactly
when a bigger model starts to pay off.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from .config import Config
from .dataset import SPLIT_NAMES, build_bundle, load_ensembles, split_proteins
from .evaluate import evaluate_split
from .prepare import discover_pdb_files, write_json
from .train import resolve_device, train_model

REGISTRY_NAME = "rounds.jsonl"
REGISTRY_MD = "rounds.md"
SPLIT_NAME = "split.json"


# --------------------------------------------------------------------------- #
# Frozen split
# --------------------------------------------------------------------------- #


def freeze_split(
    ensembles: dict, cfg: Config, path: str, *, verbose: bool = True
) -> dict[str, list[str]]:
    """Return the protein split, computing it once and reusing it forever.

    Any protein that appears after the split was frozen goes into **train**.  That
    is the whole point: the test set must not move, otherwise "test MAE improved"
    could just mean "a hard protein left the test set".
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            splits = json.load(fh)
        known = {p for ids in splits.values() for p in ids}
        added = [p for p in sorted(ensembles) if p not in known]
        if added:
            splits.setdefault("train", []).extend(added)
            write_json(path, splits)
            if verbose:
                print(f"  split is frozen; {len(added)} new protein(s) -> train: {added}",
                      flush=True)
        return {k: list(splits.get(k, [])) for k in SPLIT_NAMES}

    splits = split_proteins(
        sorted(ensembles), seed=cfg.train.seed,
        val_fraction=cfg.train.val_fraction, test_fraction=cfg.train.test_fraction,
    )
    write_json(path, splits)
    if verbose:
        print(f"  froze the split for every later round: "
              f"train={len(splits['train'])} val={len(splits['val'])} "
              f"test={len(splits['test'])}", flush=True)
    return {k: list(splits.get(k, [])) for k in SPLIT_NAMES}


def apply_split(bundle, splits: dict[str, list[str]]):
    """Rebuild ``rows``/``y_raw``/normalisation from an explicit protein split.

    ``build_bundle`` always derives the split itself, so this overwrites it.  The
    normalisation has to be recomputed here too, because it depends on which
    samples ended up in train.
    """
    rows: dict[str, list[tuple[str, int]]] = {k: [] for k in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        for protein in splits.get(split, []):
            if protein in bundle.ensembles:
                rows[split].extend((protein, f) for f in range(len(bundle.ensembles[protein])))

    bundle.rows = rows
    bundle.y_raw = {
        split: np.asarray([bundle.raw_target(p, f) for p, f in rows[split]], dtype=np.float64)
        for split in SPLIT_NAMES
    }
    train_y = bundle.y_raw["train"]
    bundle.norm_mean = float(np.mean(train_y)) if train_y.size else 0.0
    bundle.norm_std = float(np.std(train_y)) if train_y.size else 1.0
    if not np.isfinite(bundle.norm_std) or bundle.norm_std < 1e-8:
        bundle.norm_std = 1.0
    bundle.protein_splits = {k: list(splits.get(k, [])) for k in SPLIT_NAMES}
    return bundle


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@dataclass
class RoundRecord:
    """Everything worth knowing about one training round."""

    round: int
    n_proteins: int
    n_train: int
    n_val: int
    n_test: int
    val_mae: float
    test_mae: float
    test_rmse: float
    test_r2: float
    test_spearman: float
    within_protein_rho: float
    constant_baseline_mae: float
    prediction_span_ratio: float
    n_parameters: int
    best_epoch: int
    epochs_run: int
    norm_mean: float
    norm_std: float
    warm_started_from: str | None
    wall_time_s: float
    checkpoint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IterationRegistry:
    """Append-only JSONL log of rounds, plus a rendered Markdown table."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.jsonl = os.path.join(out_dir, REGISTRY_NAME)
        self.markdown = os.path.join(out_dir, REGISTRY_MD)

    def load(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.jsonl):
            return []
        records = []
        with open(self.jsonl, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def append(self, record: RoundRecord) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        # jsonutil: a round's metrics can be NaN (e.g. no protein had enough
        # frames for a rank correlation); the bare NaN token is not valid JSON.
        from .jsonutil import dumps_json

        with open(self.jsonl, "a", encoding="utf-8") as fh:
            fh.write(dumps_json(record.to_dict()) + "\n")
        self.render()

    def best(self, key: str = "val_mae") -> dict[str, Any] | None:
        records = [r for r in self.load() if np.isfinite(r.get(key, float("nan")))]
        return min(records, key=lambda r: r[key]) if records else None

    def render(self) -> str:
        records = self.load()
        if not records:
            return ""
        header = (
            "| round | proteins | train/val/test | val MAE | test MAE | test RMSE | "
            "test R² | test ρ | 组内 ρ | 常数基线 | 预测跨度 | 最佳轮 | 用时 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        )
        lines = []
        for r in records:
            lines.append(
                f"| {r['round']} | {r['n_proteins']} | "
                f"{r['n_train']}/{r['n_val']}/{r['n_test']} | "
                f"{r['val_mae']:.2f} | {r['test_mae']:.2f} | {r['test_rmse']:.2f} | "
                f"{r['test_r2']:.4f} | {r['test_spearman']:.4f} | "
                f"{r['within_protein_rho']:.4f} | {r['constant_baseline_mae']:.2f} | "
                f"{100 * r['prediction_span_ratio']:.0f}% | {r['best_epoch']} | "
                f"{r['wall_time_s']:.0f}s |"
            )
        text = (
            "# 迭代训练记录\n\n"
            "每次 `python -m pdbenergy.iterate` 追加一行。切分在第 1 轮冻结，"
            "之后新增的蛋白只进 train，所以 test 指标跨轮可比。\n\n"
            + header + "\n".join(lines) + "\n\n"
            "> 选模型只看 **val MAE**；test 指标只用于报告，不参与选择。\n"
            "> 「预测跨度」= (预测最大值 − 最小值) / (真实最大值 − 最小值)，"
            "越接近 100% 说明能量范围学得越完整。\n"
        )
        with open(self.markdown, "w", encoding="utf-8") as fh:
            fh.write(text)
        return text


# --------------------------------------------------------------------------- #
# One round
# --------------------------------------------------------------------------- #


def _prediction_span_ratio(report: dict) -> float:
    rng = report.get("energy_range", {})
    true_span = (rng.get("true_max") or 0) - (rng.get("true_min") or 0)
    pred_span = (rng.get("pred_max") or 0) - (rng.get("pred_min") or 0)
    return float(pred_span / true_span) if true_span else float("nan")


def _new_pdb_files(
    raw_dir: str,
    interim_dir: str,
    *,
    max_residues: int,
    min_residues: int,
    verbose: bool = True,
) -> dict[str, str]:
    """PDB files in ``data/raw`` that are worth labelling and have no ensemble yet.

    The residue bounds are applied **here**, before anything reaches OpenMM.  This
    is the difference between a job that finishes and one that silently tries to
    parameterise a 9000-residue assembly: parsing 2741 files takes a couple of
    minutes, whereas discovering the problem halfway through the physics would
    cost days.
    """
    from .prepare import select_usable_entries

    selected, reports = select_usable_entries(
        raw_dir, max_residues=max_residues, min_residues=min_residues,
        already_labelled_dir=interim_dir,
    )
    if verbose and reports:
        import collections

        rejected = collections.Counter(
            r.reason.split(" exceeds")[0].split(" is below")[0] for r in reports
            if not r.is_usable
        )
        if rejected:
            summary = ", ".join(f"{n} x {why}" for why, n in rejected.most_common(4))
            print(f"  filtered out {sum(rejected.values())} entry/entries: {summary}",
                  flush=True)
    return selected


#: Measured on the 14-protein set: ~0.5 MB of cached graph per conformation
#: (edge index as int64 dominates).  Used only to *predict* the footprint.
BYTES_PER_CACHED_FRAME = 0.5 * 1024 * 1024


def should_cache(interim_dir: str, budget_gb: float) -> tuple[bool, str]:
    """Decide whether the graph cache can fit, from the labelled data on disk.

    The cache holds every graph in RAM as one dict, so it scales linearly with
    the number of conformations.  It is a large win (featurisation is ~150 ms per
    frame, paid *every epoch*), but past a few tens of thousands of frames it is
    simply an out-of-memory error waiting to happen - and it fails only *after*
    spending hours building the cache.  So the decision is made up front.
    """
    if not os.path.isdir(interim_dir):
        return True, "no labelled data yet"
    frames = 0
    for name in os.listdir(interim_dir):
        if not name.endswith(".npz"):
            continue
        try:
            with np.load(os.path.join(interim_dir, name)) as data:
                frames += int(data["coords"].shape[0])
        except Exception:
            continue
    if frames == 0:
        return True, "no labelled data yet"
    estimated_gb = frames * BYTES_PER_CACHED_FRAME / (1024 ** 3)
    if estimated_gb > budget_gb:
        return False, (
            f"{frames:,} conformations would need about {estimated_gb:.0f} GB of RAM "
            f"(budget {budget_gb:.0f} GB). Featurising on the fly instead; add "
            f"--num-workers to parallelise it, or raise --cache-budget if you really "
            f"have the memory."
        )
    return True, f"{frames:,} conformations, about {estimated_gb:.1f} GB of RAM"


def run_rounds(
    cfg: Config,
    *,
    rounds: int = 1,
    out_dir: str = "outputs/iterative",
    interim_dir: str = "data/interim",
    raw_dir: str = "data/raw",
    build_new: bool = True,
    warm_start: bool = True,
    build_limit: int = 0,
    threads: int = 8,
    workers: int = 1,
    verbose: bool = True,
) -> list[RoundRecord]:
    """Run ``rounds`` training rounds on top of each other."""
    os.makedirs(out_dir, exist_ok=True)
    registry = IterationRegistry(out_dir)
    existing = registry.load()
    starting_round = (max(r["round"] for r in existing) + 1) if existing else 1
    split_path = os.path.join(out_dir, SPLIT_NAME)

    produced: list[RoundRecord] = []
    for offset in range(rounds):
        round_index = starting_round + offset
        print(f"\n{'=' * 74}\n== round {round_index}\n{'=' * 74}", flush=True)

        # ---- 1. label any newly added structures -------------------------- #
        if build_new:
            pending = _new_pdb_files(
                raw_dir, interim_dir,
                max_residues=cfg.prepare.max_residues,
                min_residues=cfg.prepare.min_residues,
                verbose=verbose,
            )
            if pending:
                from .ensemble import build_all

                if build_limit and len(pending) > build_limit:
                    # Deterministic subset, so successive calls walk the corpus in
                    # a stable order and never re-label the same entries.
                    chosen = dict(sorted(pending.items())[:build_limit])
                    print(f"  staging: labelling {len(chosen)} of {len(pending)} pending "
                          f"entries this round (--build-limit {build_limit})", flush=True)
                else:
                    chosen = pending
                print(f"  labelling {len(chosen)} entry/entries "
                      f"(this is the expensive step)", flush=True)
                build_all(
                    chosen, out_dir=interim_dir,
                    ensemble_cfg=cfg.ensemble, label_cfg=cfg.label,
                    prepare_cfg=cfg.prepare, threads=threads, workers=workers,
                    overwrite=False, verbose=True,
                )
            elif verbose:
                print("  no new labelable entries; training on what is already labelled",
                      flush=True)

        # ---- 2. dataset with the frozen split ----------------------------- #
        ensembles = load_ensembles(interim_dir, verbose=False)
        bundle = build_bundle(ensembles, cfg, verbose=False)
        splits = freeze_split(ensembles, cfg, split_path, verbose=verbose)
        apply_split(bundle, splits)
        print(
            f"  {len(ensembles)} proteins, {len(bundle.all_rows())} conformers | "
            f"train={len(bundle.rows['train'])} val={len(bundle.rows['val'])} "
            f"test={len(bundle.rows['test'])}", flush=True,
        )

        # ---- 3. train (warm-started from the previous round) -------------- #
        round_dir = os.path.join(out_dir, f"round_{round_index:03d}")
        warm_from = None
        if warm_start and existing:
            candidates = [r["checkpoint"] for r in existing if os.path.exists(r.get("checkpoint", ""))]
            if candidates:
                warm_from = candidates[-1]
        result = train_model(
            bundle,
            descriptor_mode=(cfg.model.kind == "mlp"),
            out_dir=round_dir,
            verbose=verbose,
            warm_start_from=warm_from,
        )

        # ---- 4. evaluate on the frozen splits ----------------------------- #
        descriptor_mode = cfg.model.kind == "mlp"
        device = resolve_device(cfg.train.device)
        val_report = evaluate_split(result.model, bundle, "val", descriptor_mode=descriptor_mode,
                                    device=device)
        test_report = evaluate_split(result.model, bundle, "test", descriptor_mode=descriptor_mode,
                                     device=device)
        test_overall = test_report["overall"]
        test_rank = test_report["rank_discrimination"]

        # Constant-prediction baseline for THIS round's train statistics.
        from .evaluate import constant_baseline

        baseline = constant_baseline(bundle, "test").get("mae", float("nan"))

        record = RoundRecord(
            round=round_index,
            n_proteins=len(ensembles),
            n_train=len(bundle.rows["train"]),
            n_val=len(bundle.rows["val"]),
            n_test=len(bundle.rows["test"]),
            val_mae=float(val_report["overall"]["mae"]),
            test_mae=float(test_overall["mae"]),
            test_rmse=float(test_overall["rmse"]),
            test_r2=float(test_overall["r2"]),
            test_spearman=float(test_overall["spearman"]),
            within_protein_rho=float(test_rank["mean_spearman"]),
            constant_baseline_mae=float(baseline),
            prediction_span_ratio=_prediction_span_ratio(test_report),
            n_parameters=result.n_parameters,
            best_epoch=result.best_epoch,
            epochs_run=int(result.history["epoch"][-1]) if result.history.get("epoch") else 0,
            norm_mean=float(bundle.norm_mean),
            norm_std=float(bundle.norm_std),
            warm_started_from=warm_from,
            wall_time_s=float(result.wall_time_s),
            checkpoint=os.path.join(round_dir, "checkpoint.pt"),
        )
        registry.append(record)
        produced.append(record)
        existing = registry.load()

        # ---- 5. keep the best-by-validation checkpoint -------------------- #
        best = registry.best("val_mae")
        if best and os.path.abspath(best["checkpoint"]) == os.path.abspath(record.checkpoint):
            best_dir = os.path.join(out_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            shutil.copy2(record.checkpoint, os.path.join(best_dir, "checkpoint.pt"))
            for name in ("config.json", "train_state.pt"):
                src = os.path.join(round_dir, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(best_dir, name))
            print(f"  new best by validation MAE ({record.val_mae:.3f}) -> {best_dir}", flush=True)

        print(
            f"\n  round {round_index}: val MAE {record.val_mae:.2f} | "
            f"test MAE {record.test_mae:.2f} (baseline {record.constant_baseline_mae:.2f}) | "
            f"test ρ {record.test_spearman:.3f} | span {100 * record.prediction_span_ratio:.0f}%",
            flush=True,
        )

        # ---- 6. honest advisory when the model is now the bottleneck ------ #
        if record.n_train > 5000 and cfg.model.hidden_dim <= 64:
            print(
                f"  NOTE: {record.n_train} training conformers is a lot for "
                f"hidden_dim={cfg.model.hidden_dim}. The model may now be the limit -\n"
                f"        consider a new run with --hidden-dim 128 --interactions 4 "
                f"(warm start cannot load a different architecture, so train from scratch).",
                flush=True,
            )

    print()
    print(registry.render())
    return produced


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pdbenergy.iterate",
        description="Iteratively retrain as you add PDB structures, with a frozen test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rounds", type=int, default=1, help="how many training rounds to run")
    parser.add_argument("--out-dir", default="outputs/iterative")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--config", default=None, help="JSON config to use as the base")
    parser.add_argument("--epochs", type=int, default=0, help="epochs per round (0 = config default)")
    parser.add_argument("--model", default=None, choices=["schnet", "mlp"])
    # BooleanOptionalAction gives BOTH spellings: --build-new / --no-build-new.
    # The documentation said --build-new while the parser only accepted
    # --no-build-new, so following the docs raised "unrecognized arguments".
    parser.add_argument("--build-new", action=argparse.BooleanOptionalAction, default=True,
                        help="label PDB files in --raw-dir that have no ensemble yet "
                             "(use --no-build-new to skip)")
    parser.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True,
                        help="start each round from the previous round's checkpoint "
                             "(use --no-warm-start for the from-scratch ablation)")
    parser.add_argument("--cache-graphs", action=argparse.BooleanOptionalAction, default=None,
                        help="cache featurised graphs in RAM. Default: on, but automatically "
                             "disabled when the dataset is too large to fit (see --cache-budget)")
    parser.add_argument("--cache-budget", type=float, default=20.0, metavar="GB",
                        help="disable the graph cache above this estimated memory footprint")
    parser.add_argument("--max-residues", type=int, default=None,
                        help="only label entries up to this size (default: config, 120)")
    parser.add_argument("--build-limit", type=int, default=0, metavar="N",
                        help="label at most N new entries this round (0 = all pending). "
                             "Use this to stage a large corpus: each call is resumable, "
                             "so you can stop, inspect and continue")
    parser.add_argument("--min-residues", type=int, default=None,
                        help="skip fragments shorter than this (default: config, 10)")
    parser.add_argument("--threads", type=int, default=8, help="OpenMM threads per ensemble worker")
    parser.add_argument("--workers", type=int, default=1, help="parallel ensemble workers")
    parser.add_argument("--report", action="store_true",
                        help="only print the accumulated round table, then exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config) if args.config else Config()
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.model:
        cfg.model.kind = args.model
    if args.max_residues is not None:
        cfg.prepare.max_residues = args.max_residues
    if args.min_residues is not None:
        cfg.prepare.min_residues = args.min_residues
    if args.cache_graphs is not None:
        cfg.train.cache_graphs = bool(args.cache_graphs)
    if cfg.train.cache_graphs:
        # Refuse to build the cache if it cannot possibly fit: featurising 100k
        # frames takes hours and then dies on allocation.
        keep, reason = should_cache(args.interim_dir, args.cache_budget)
        if not keep:
            print(f"  graph cache disabled: {reason}", flush=True)
            cfg.train.cache_graphs = False

    if args.report:
        registry = IterationRegistry(args.out_dir)
        text = registry.render()
        if not text:
            print(f"no rounds recorded yet in {args.out_dir!r}")
            return 1
        print(text)
        return 0

    run_rounds(
        cfg,
        rounds=args.rounds,
        out_dir=args.out_dir,
        interim_dir=args.interim_dir,
        raw_dir=args.raw_dir,
        build_new=args.build_new,
        build_limit=args.build_limit,
        warm_start=args.warm_start,
        threads=args.threads,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
