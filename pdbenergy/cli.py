"""Command-line interface: the whole pipeline, one step at a time.

    python -m pdbenergy.cli download            # fetch PDB entries from RCSB
    python -m pdbenergy.cli inventory           # what did we actually get?
    python -m pdbenergy.cli ensemble            # label conformations with physics
    python -m pdbenergy.cli dataset             # splits + normalisation stats
    python -m pdbenergy.cli train               # fit the graph neural network
    python -m pdbenergy.cli evaluate            # metrics + figures
    python -m pdbenergy.cli predict x.pdb       # energy for your own structure
    python -m pdbenergy.cli ablate              # measure the split-leakage effect
    python -m pdbenergy.cli all                 # everything, end to end

Every step writes its artifacts under ``data/`` and ``outputs/`` and can be
re-run safely: nothing is recomputed unless asked.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

from .config import (
    Config,
    DEFAULT_PDB_IDS,
    EnsembleConfig,
    LabelConfig,
    PrepareConfig,
)


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #

#: Speed/accuracy trade-offs for the physics step.  ``default`` is what the
#: shipped results use.  ``quick`` is for a smoke test on a laptop, ``thorough``
#: for a real study (more sampling, exact non-bonded treatment).
PRESETS: dict[str, dict] = {
    "quick": {
        "label": {"nonbonded_method": "CutoffNonPeriodic", "cutoff_nm": 1.0},
        "ensemble": {
            "torsion_levels": (1.0, 2.0),
            "n_torsion_per_level": 8,
            "minimise_fraction": 0.1,
            "partial_minimise_iterations": 20,
            "reference_minimise_iterations": 80,
            "md_temperatures": (300.0,),
            "md_ps_per_temperature": 0.4,
            "md_save_every_ps": 0.2,
            "use_nmr_models": False,
            "max_nmr_models": 0,
        },
    },
    "default": {},
    "thorough": {
        "label": {"nonbonded_method": "NoCutoff"},
        "ensemble": {
            "torsion_levels": (1.0, 1.5, 2.0, 2.5, 3.0),
            "n_torsion_per_level": 24,
            "minimise_fraction": 0.2,
            "md_temperatures": (280.0, 320.0, 380.0, 450.0, 550.0),
            "md_ps_per_temperature": 2.0,
            "md_save_every_ps": 0.2,
            "max_nmr_models": 20,
        },
    },
}


def apply_preset(cfg: Config, preset: str) -> Config:
    """Return a config with a named preset applied (non-mutating)."""
    if preset not in PRESETS:
        raise SystemExit(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    for section, overrides in PRESETS[preset].items():
        current = getattr(cfg, section)
        setattr(cfg, section, replace(current, **overrides))
    return cfg


def base_config(args) -> Config:
    """Load ``--config`` (if given), apply the named preset, then CLI overrides.

    Precedence, lowest to highest: dataclass defaults -> JSON file -> preset ->
    explicit command-line flags.  Every run writes the resolved config next to
    its checkpoint, so a result is always reproducible from the artifacts.
    """
    path = getattr(args, "config", None)
    cfg = Config.load(path) if path else Config()
    return apply_preset(cfg, getattr(args, "preset", "default"))


def resolve_ids(args) -> tuple[str, ...]:
    if getattr(args, "ids", None):
        return tuple(i.strip().upper() for i in args.ids if i.strip())
    return tuple(i.strip().upper() for i in DEFAULT_PDB_IDS)


# --------------------------------------------------------------------------- #
# Step implementations
# --------------------------------------------------------------------------- #


def cmd_download(args) -> int:
    from .prepare import download_many

    ids = resolve_ids(args)
    print(f"downloading {len(ids)} entries into {args.raw_dir}")
    paths = download_many(ids, args.raw_dir, force=args.force)
    print(f"downloaded {len(paths)}/{len(ids)}")
    return 0 if paths else 1


def cmd_inventory(args) -> int:
    from .prepare import inventory, print_inventory, write_json

    reports = inventory(
        args.raw_dir, max_residues=args.max_residues, min_residues=args.min_residues
    )
    if not reports:
        print(f"no PDB files in {args.raw_dir!r}; run `download` first")
        return 1

    usable = [r for r in reports if r.is_usable]
    rejected = [r for r in reports if not r.is_usable]

    # Printing 2700 rows helps nobody; below ~60 entries the detail is worth it.
    if len(reports) <= 60 and not args.no_table:
        print_inventory(reports)
    elif args.no_table:
        pass

    if rejected:
        import collections

        def bucket(reason: str) -> str:
            for marker in (" exceeds", " is below", " (multi-chain"):
                if marker in reason:
                    head, _, tail = reason.partition(marker)
                    return f"{head}{marker.rstrip()}{tail.split('=')[-1] if '=' in tail else ''}"
            return reason

        counts = collections.Counter(bucket(r.reason) for r in rejected)
        print(f"\n排除原因（共 {len(rejected)} 条）：")
        for reason, n in counts.most_common(8):
            print(f"  {n:>5}  {reason}")

    over_cap = [r for r in rejected if "max_residues" in r.reason]
    if over_cap:
        print(
            f"\n其中 {len(over_cap)} 条只是被 max_residues={args.max_residues} 挡住。"
            f"\n放宽上限可多收："
        )
        for cap in (200, 300, 500):
            extra = sum(1 for r in over_cap if r.n_residues <= cap)
            print(f"  max_residues={cap:<4} -> 再多 {extra} 条")

    total_res = sorted(r.n_residues for r in usable)
    if total_res:
        print(
            f"\n{len(usable)}/{len(reports)} 条可用"
            f"（残基数 中位数 {total_res[len(total_res)//2]}，"
            f"p95 {total_res[int(0.95 * (len(total_res) - 1))]}，最大 {total_res[-1]}）"
        )
    else:
        print(f"\n0/{len(reports)} 条可用")

    write_json(
        os.path.join(args.outputs_dir, "inventory.json"),
        [r.to_dict() for r in reports],
    )
    print(f"完整报告 -> {os.path.join(args.outputs_dir, 'inventory.json')}")
    return 0


def cmd_ensemble(args) -> int:
    from .ensemble import build_all
    from .prepare import discover_pdb_files

    cfg = base_config(args)
    if args.limit:
        cfg.ensemble.torsion_levels = cfg.ensemble.torsion_levels[: max(1, args.limit)]
    paths = discover_pdb_files(args.raw_dir)
    if not paths:
        print(f"no PDB files in {args.raw_dir!r}; run `download` first")
        return 1
    wanted = getattr(args, "ids", None)
    if wanted:
        keep = {i.strip().upper() for i in wanted if i.strip()}
        missing = sorted(keep - set(paths))
        if missing:
            print(f"warning: no raw file for {missing}")
        paths = {k: v for k, v in paths.items() if k in keep}
        if not paths:
            print("no proteins left after --ids filter")
            return 1
    print(
        f"physics: {' + '.join(cfg.label.forcefield_files)} | "
        f"solvent={cfg.label.implicit_solvent or 'vacuum'} | "
        f"{cfg.label.nonbonded_method} {cfg.label.cutoff_nm} nm | "
        f"threads={args.threads} workers={args.workers} | "
        f"{len(paths)} proteins"
    )
    written = build_all(
        paths,
        out_dir=args.interim_dir,
        ensemble_cfg=cfg.ensemble,
        label_cfg=cfg.label,
        prepare_cfg=cfg.prepare,
        threads=args.threads,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(f"\nwrote {len(written)} ensembles to {args.interim_dir}")
    return 0 if written else 1


def cmd_dataset(args) -> int:
    from .dataset import build_bundle, load_ensembles
    from .prepare import write_json

    cfg = base_config(args)
    if args.split_mode:
        cfg.train.split_mode = args.split_mode
    if args.target:
        cfg.train.target = args.target
    ensembles = load_ensembles(args.interim_dir)
    bundle = build_bundle(ensembles, cfg)
    out = os.path.join(args.processed_dir, "dataset")
    os.makedirs(out, exist_ok=True)
    bundle.save_rows(os.path.join(out, "splits.json"))
    write_json(os.path.join(out, "summary.json"), bundle.summary())
    print(f"\nwrote {out}/splits.json and summary.json")
    return 0


def cmd_train(args) -> int:
    from .dataset import build_bundle, load_ensembles
    from .train import train_model, resolve_device

    cfg = base_config(args)
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.learning_rate:
        cfg.train.learning_rate = args.learning_rate
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.split_mode:
        cfg.train.split_mode = args.split_mode
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    cfg.model.kind = args.model
    if args.hidden_dim:
        cfg.model.hidden_dim = args.hidden_dim
    if args.interactions:
        cfg.model.n_interactions = args.interactions
    cfg.train.device = args.device
    if getattr(args, "no_cache_graphs", False):
        cfg.train.cache_graphs = False

    ensembles = load_ensembles(args.interim_dir)
    bundle = build_bundle(ensembles, cfg)
    tag = args.tag or f"{args.model}_{cfg.train.split_mode}"
    run_dir = os.path.join(args.outputs_dir, tag)
    print(f"\ntraining {args.model} -> {run_dir}  (device={resolve_device(args.device)})")
    result = train_model(
        bundle,
        descriptor_mode=(args.model == "mlp"),
        out_dir=run_dir,
        warm_start_from=getattr(args, "init_from", None),
        resume_from=getattr(args, "resume", None),
        recompute_normalisation=getattr(args, "recompute_normalisation", False),
    )
    print(
        f"\nbest epoch {result.best_epoch} | best val MAE {result.best_val_mae:.3f} kcal/mol | "
        f"{result.n_parameters:,} params | {result.wall_time_s:.1f} s"
    )
    print(f"checkpoint -> {os.path.join(run_dir, 'checkpoint.pt')}")
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import evaluate_run
    from .prepare import write_json

    report = evaluate_run(
        args.run_dir,
        interim_dir=args.interim_dir,
        out_dir=args.eval_dir or None,
    )
    write_json(os.path.join(args.run_dir, "eval", "metrics_full.json"), report)
    print("\nmetrics (MAE/RMSE in kcal/mol; 'constant' = always predict the train mean):")
    print(f"  {'split':>5} {'n':>5} {'MAE':>9} {'RMSE':>9} {'R2':>8} {'rho':>8} "
          f"{'rank-rho':>9} {'constant':>9}")
    for split, data in report["splits"].items():
        m = data["overall"]
        rank = data.get("rank_discrimination", {})
        base = report.get("baselines", {}).get(split, {})
        const = f"{base['mae']:9.3f}" if base else " " * 9
        print(
            f"  {split:>5} {m['n']:>5} {m['mae']:9.3f} {m['rmse']:9.3f} {m['r2']:8.4f} "
            f"{m['spearman']:8.4f} "
            f"{rank.get('mean_spearman', float('nan')):9.4f} {const}"
        )
    return 0


def cmd_predict(args) -> int:
    from .prepare import write_json
    from .predict import EnergyPredictor

    checkpoint = args.checkpoint or os.path.join(args.outputs_dir, "schnet_protein", "checkpoint.pt")
    if not os.path.exists(checkpoint):
        print(f"no checkpoint at {checkpoint!r}; train a model first")
        return 1
    predictor = EnergyPredictor(checkpoint, device=args.device, threads=args.threads)

    paths = list(args.pdb)
    if not paths:
        print("no PDB file given")
        return 1
    rows: list[dict] = []
    for path in paths:
        print(f"\n{path}")
        for pred in predictor.predict_file(
            path, max_models=args.max_models, verify=args.verify, verbose=not args.quiet
        ):
            row = pred.to_dict()
            row["path"] = path
            rows.append(row)

    if len(rows) > 1:
        ranked = sorted(rows, key=lambda r: r["predicted_relative_energy_kcal_per_mol"])
        print("\nranking (most favourable first):")
        for i, row in enumerate(ranked, start=1):
            extra = ""
            if row.get("true_relative_energy_kcal_per_mol") is not None:
                extra = f" | true {row['true_relative_energy_kcal_per_mol']:8.3f}"
            print(
                f"  {i:>2}. {os.path.basename(row['path']):<16} model "
                f"{row['model_index']:<3} {row['predicted_relative_energy_kcal_per_mol']:9.3f} "
                f"kcal/mol{extra}"
            )
    if args.json:
        write_json(args.json, rows)
        print(f"\nwrote {args.json}")
    return 0


def cmd_ablate(args) -> int:
    """Train the same model twice, changing only the *split*, and compare.

    The gap between a leaky frame-level split and an honest protein-level split
    is a direct measurement of how much data leakage inflates apparent accuracy.

    The ablation defaults to the cheap descriptor-MLP arm.  Leakage is a property
    of the *data*, not of the architecture, so a small fast model measures it just
    as well - and it makes the experiment take minutes instead of hours.  Use
    ``--model schnet`` to reproduce it with the graph network.
    """
    from torch.utils.data import DataLoader

    from .dataset import build_bundle, collate_for, load_ensembles
    from .prepare import write_json
    from .train import evaluate_loader, regression_metrics, resolve_device, train_model

    model_kind = getattr(args, "model", "mlp")
    descriptor_mode = model_kind == "mlp"

    results: dict[str, dict] = {}
    for mode in ("protein", "frame"):
        cfg = base_config(args)
        cfg.model.kind = model_kind
        cfg.train.split_mode = mode
        cfg.train.epochs = args.epochs
        cfg.train.seed = args.seed
        ensembles = load_ensembles(args.interim_dir, verbose=False)
        bundle = build_bundle(ensembles, cfg, verbose=False)
        run_dir = os.path.join(args.outputs_dir, f"ablation_{mode}")
        print(f"\n=== split_mode={mode} model={model_kind} -> {run_dir} ===")
        result = train_model(bundle, descriptor_mode=descriptor_mode, out_dir=run_dir)
        device = resolve_device(cfg.train.device)
        test_ds = bundle.dataset("test", descriptor_mode=descriptor_mode)
        loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            collate_fn=collate_for(descriptor_mode))
        _, y_true_n, y_pred_n = evaluate_loader(result.model, loader, device, descriptor_mode)
        m = regression_metrics(bundle.denormalise(y_true_n), bundle.denormalise(y_pred_n))
        results[mode] = {
            "model": model_kind,
            "test_mae": m.mae,
            "test_rmse": m.rmse,
            "test_r2": m.r2,
            "test_spearman": m.spearman,
            "best_val_mae": result.best_val_mae,
            "run_dir": run_dir,
        }
        print(f"  -> test MAE {m.mae:.3f} kcal/mol, R2 {m.r2:.4f}")

    if "protein" in results and "frame" in results:
        factor = results["protein"]["test_mae"] / max(1e-9, results["frame"]["test_mae"])
        results["leakage_inflation_factor"] = factor
        print(
            f"\nframe-level split reports {factor:.2f}x the test MAE "
            f"({results['frame']['test_mae']:.3f} vs {results['protein']['test_mae']:.3f} kcal/mol)."
        )
        print(
            "Caveat: the size of that gap depends on how much capacity the model has\n"
            "to exploit the leakage - a small model can barely memorise frames.  The\n"
            "direct measurement below does not depend on the model at all."
        )

    # Model-free measurement of the actual overlap between the splits.
    try:
        from .leakage import compare_split_modes

        print("\n-- direct leakage measurement (no model involved) --")
        cfg = base_config(args)
        ensembles = load_ensembles(args.interim_dir, verbose=False)
        results["direct_leakage"] = compare_split_modes(ensembles, cfg)
        same = results["direct_leakage"].get("frame", {}).get("same_protein_fraction")
        if same is not None:
            print(
                f"\nUnder the frame-level split, {100 * same:.0f}% of test frames have a "
                "training\nneighbour from the SAME protein (near-duplicate structures with "
                "nearly equal\nenergies). Under the protein-level split that number is "
                "0% by construction.\nThat - not the MAE gap - is what data leakage is."
            )
    except Exception as exc:  # measurement must never break the ablation
        print(f"  (direct leakage measurement skipped: {type(exc).__name__}: {exc})")
        results["direct_leakage"] = {}
    write_json(os.path.join(args.outputs_dir, "ablation", "split_leakage.json"), results)
    return 0


def cmd_all(args) -> int:
    """Run the full pipeline in the order that matters."""
    cfg = base_config(args)
    steps = [
        (cmd_download, "download"),
        (cmd_inventory, "inventory"),
        (cmd_ensemble, "ensemble"),
        (cmd_dataset, "dataset"),
    ]
    for fn, name in steps:
        print(f"\n{'=' * 78}\n== {name}\n{'=' * 78}")
        code = fn(args)
        if code != 0:
            print(f"step {name} failed (exit {code})")
            return code
    for model in ("schnet", "mlp"):
        args.model = model
        args.tag = f"{model}_protein"
        print(f"\n{'=' * 78}\n== train {model}\n{'=' * 78}")
        cmd_train(args)
    for model in ("schnet", "mlp"):
        args.run_dir = os.path.join(args.outputs_dir, f"{model}_protein")
        args.eval_dir = None
        print(f"\n{'=' * 78}\n== evaluate {model}\n{'=' * 78}")
        cmd_evaluate(args)
    print(f"\n{'=' * 78}\n== ablation: split leakage\n{'=' * 78}")
    cmd_ablate(args)
    print("\nDone. See outputs/ for checkpoints, metrics.json and figures.")
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdbenergy",
        description="Learn a neural-network surrogate for protein conformational energy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--preset", default="default", choices=sorted(PRESETS))
    parser.add_argument(
        "--config", default=None,
        help="JSON config file used as the base (see configs/default.json)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch PDB entries from RCSB")
    p.add_argument("--ids", nargs="*", help="PDB IDs (default: the built-in list)")
    p.add_argument("--force", action="store_true", help="re-download existing files")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("inventory", help="report what is inside data/raw")
    p.add_argument("--max-residues", type=int, default=120)
    p.add_argument("--min-residues", type=int, default=10)
    p.add_argument("--no-table", action="store_true",
                   help="skip the per-entry table (useful for thousands of files)")
    p.set_defaults(func=cmd_inventory)

    p = sub.add_parser("ensemble", help="generate and physics-label conformations")
    p.add_argument("--threads", type=int, default=8, help="OpenMM threads per worker")
    p.add_argument("--workers", type=int, default=1, help="proteins processed in parallel")
    p.add_argument("--limit", type=int, default=0, help="cap the number of torsion levels")
    p.add_argument("--ids", nargs="*", default=None,
                   help="restrict to these PDB IDs (default: every file in --raw-dir)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_ensemble)

    p = sub.add_parser("dataset", help="build splits and normalisation statistics")
    p.add_argument("--split-mode", choices=["protein", "frame"])
    p.add_argument("--target", choices=["relative", "absolute"])
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("train", help="train the energy model")
    p.add_argument("--model", default="schnet", choices=["schnet", "mlp"])
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=0)
    p.add_argument("--interactions", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--split-mode", choices=["protein", "frame"])
    p.add_argument("--device", default="auto")
    p.add_argument("--tag", default=None, help="output subdirectory name")
    p.add_argument("--init-from", default=None, metavar="CHECKPOINT",
                   help="warm-start from this checkpoint.pt (fresh optimiser/history); "
                        "use when you have MORE DATA and want to improve on a previous model")
    p.add_argument("--resume", default=None, metavar="RUN_DIR",
                   help="continue an interrupted run: restores weights, optimiser state "
                        "and the epoch counter from that run directory")
    p.add_argument("--recompute-normalisation", action="store_true",
                   help="with --init-from/--resume, recompute the target mean/std from the "
                        "new training split instead of reusing the checkpoint's")
    p.add_argument("--no-cache-graphs", action="store_true",
                   help="featurise on the fly instead of caching all graphs in RAM "
                        "(needed once the dataset exceeds ~20k frames)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="metrics and figures for a trained run")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--eval-dir", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="predict energies for PDB files")
    p.add_argument("pdb", nargs="*", help="one or more PDB files")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--max-models", type=int, default=1)
    p.add_argument("--verify", action="store_true", help="also compute the true energy")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--json", default=None, help="write predictions to this JSON file")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("ablate", help="measure train/test split leakage")
    p.add_argument("--epochs", type=int, default=20,
                   help="same budget for both arms, so the comparison is fair")
    p.add_argument("--model", default="mlp", choices=["mlp", "schnet"],
                   help="leakage is a property of the data; the cheap arm measures it fine")
    p.add_argument("--seed", type=int, default=1234)
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("all", help="run the whole pipeline")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-residues", type=int, default=120)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--split-mode", choices=["protein", "frame"], default=None)
    p.add_argument("--target", choices=["relative", "absolute"], default=None)
    p.add_argument("--model", default="schnet")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=0)
    p.add_argument("--interactions", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto")
    p.add_argument("--tag", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--eval-dir", default=None)
    p.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # required for Windows multiprocessing spawn
    sys.exit(main())
