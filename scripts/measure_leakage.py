# -*- coding: utf-8 -*-
"""Measure train/test overlap directly (no training) and save the result.

    python scripts/measure_leakage.py

Writes ``outputs/ablation/direct_leakage.json``, which
``scripts/write_results.py`` folds into TeachFlow.md chapter 11.
"""

from __future__ import annotations

import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pdbenergy.config import Config                       # noqa: E402
from pdbenergy.dataset import load_ensembles              # noqa: E402
from pdbenergy.leakage import compare_split_modes         # noqa: E402


def main() -> int:
    interim = os.path.join(ROOT, "data", "interim")
    out_path = os.path.join(ROOT, "outputs", "ablation", "direct_leakage.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ensembles = load_ensembles(interim, verbose=False)
    print("measuring nearest-neighbour overlap between train and test (no model)")
    results = compare_split_modes(ensembles, Config())
    with io.open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print("wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
