"""Predict conformational energies for PDB files from Python.

    python examples/predict_pdb.py data/raw/1L2Y.pdb data/raw/1CRN.pdb
    python examples/predict_pdb.py data/raw/1L2Y.pdb --models 3 --verify

``--verify`` also computes the true AMBER14+GBn2 energy with OpenMM for the same
prepared structures, so you can see the prediction error directly instead of
taking the model's word for it.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdbenergy.predict import EnergyPredictor          # noqa: E402

DEFAULT_CHECKPOINT = os.path.join("outputs", "schnet_protein", "checkpoint.pt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdb", nargs="+", help="one or more PDB files")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--models", type=int, default=1,
                        help="how many MODEL blocks per file to score (NMR ensembles have many)")
    parser.add_argument("--verify", action="store_true",
                        help="also compute the true force-field energy")
    parser.add_argument("--threads", type=int, default=4, help="OpenMM threads for --verify")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint!r}.")
        print("train one first:  python -m pdbenergy.cli train --model schnet")
        return 1

    predictor = EnergyPredictor(args.checkpoint, device="cpu", threads=args.threads)

    rows = []
    for path in args.pdb:
        print(f"\n{path}")
        for pred in predictor.predict_file(
            path, max_models=args.models, verify=args.verify, verbose=True
        ):
            rows.append((path, pred))

    if len(rows) > 1:
        print("\nranking (most favourable first):")
        for rank, (path, pred) in enumerate(
            sorted(rows, key=lambda r: r[1].predicted_relative_energy), start=1
        ):
            extra = ""
            if pred.true_relative_energy is not None:
                extra = f" | true {pred.true_relative_energy:8.3f}"
            print(
                f"  {rank:>2}. {os.path.basename(path):<16} model {pred.model_index:<3}"
                f" {pred.predicted_relative_energy:9.3f} kcal/mol{extra}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
