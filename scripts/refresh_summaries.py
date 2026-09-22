# -*- coding: utf-8 -*-
"""Refresh data_summary.json for each run from the ensembles on disk.

Needed whenever the reporting code changes: the summary is produced at training
time, so a fix to what gets *reported* has to be back-filled without retraining.

    python scripts/refresh_summaries.py
"""

from __future__ import annotations

import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pdbenergy.config import Config                    # noqa: E402
from pdbenergy.config import FeatureConfig             # noqa: E402
from pdbenergy.dataset import build_bundle, load_ensembles  # noqa: E402


def main() -> int:
    ensembles = load_ensembles(os.path.join(ROOT, "data", "interim"), verbose=False)
    # The split must match training exactly; read it back from a checkpoint.
    import torch

    ckpt = os.path.join(ROOT, "outputs", "schnet_protein", "checkpoint.pt")
    if not os.path.exists(ckpt):
        print("no checkpoint at %s" % ckpt)
        return 1
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = Config.load(os.path.join(ROOT, "outputs", "schnet_protein", "config.json"))
    cfg.features = FeatureConfig(**payload["feature_config"])
    cfg.train.split_mode = payload.get("split_mode", "protein")
    bundle = build_bundle(ensembles, cfg, split_mode=cfg.train.split_mode, verbose=False)
    bundle.protein_splits = payload.get("protein_splits", bundle.protein_splits)

    summary = bundle.summary()
    for run in ("schnet_protein", "mlp_protein"):
        path = os.path.join(ROOT, "outputs", run, "data_summary.json")
        if not os.path.exists(os.path.dirname(path)):
            continue
        with io.open(path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print("refreshed %s" % path)
    print("residue counts now:",
          {p: v["sequence_length"] for p, v in summary["per_protein"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
