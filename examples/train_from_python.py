"""Use the library as a library: build a dataset and train, without the CLI.

This is the same path ``python -m pdbenergy.cli train`` takes, written out so the
pieces are visible and easy to modify:

    Config          every knob, one dataclass per concern
    load_ensembles  read the physics-labelled conformer ensembles
    build_bundle    protein-level split + target standardisation
    train_model     training loop, early stopping, checkpointing

    python examples/train_from_python.py --epochs 5
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdbenergy.config import Config                    # noqa: E402
from pdbenergy.dataset import build_bundle, load_ensembles   # noqa: E402
from pdbenergy.evaluate import evaluate_split          # noqa: E402
from pdbenergy.train import train_model                # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--out-dir", default="outputs/python_api_demo")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model", default="schnet", choices=["schnet", "mlp"])
    args = parser.parse_args()

    # ---- 1. configuration ------------------------------------------------- #
    cfg = Config()
    cfg.train.epochs = args.epochs
    cfg.train.log_every = 1
    cfg.model.kind = args.model
    print("resolved config:")
    print(f"  physics   {cfg.label.forcefield_files} solvent={cfg.label.implicit_solvent}")
    print(f"  features  cutoff={cfg.features.cutoff} A  n_rbf={cfg.features.n_rbf}")
    print(f"  model     {cfg.model.kind}  hidden={cfg.model.hidden_dim}  "
          f"interactions={cfg.model.n_interactions}")

    # ---- 2. data ---------------------------------------------------------- #
    ensembles = load_ensembles(args.interim_dir, verbose=False)
    bundle = build_bundle(ensembles, cfg)

    # ---- 3. train --------------------------------------------------------- #
    descriptor_mode = cfg.model.kind == "mlp"
    result = train_model(bundle, descriptor_mode=descriptor_mode, out_dir=args.out_dir)
    print(f"\nbest epoch {result.best_epoch} | best val MAE "
          f"{result.best_val_mae:.3f} kcal/mol | {result.n_parameters:,} params")

    # ---- 4. evaluate ------------------------------------------------------ #
    for split in ("val", "test"):
        if not bundle.rows.get(split):
            continue
        report = evaluate_split(result.model, bundle, split, descriptor_mode=descriptor_mode)
        m = report["overall"]
        rank = report["rank_discrimination"]
        print(f"  [{split}] n={m['n']}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
              f"R2={m['r2']:.4f}  rho={m['spearman']:.4f}  "
              f"within-protein rho={rank['mean_spearman']:.4f}")
    print(f"\ncheckpoint -> {os.path.join(args.out_dir, 'checkpoint.pt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
