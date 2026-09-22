# -*- coding: utf-8 -*-
"""Estimate the labelling workload for the PDB files currently in data/raw.

Reads ``outputs/inventory.json`` (written by ``pdbenergy inventory``) and reports,
for several residue caps, how many entries are usable and roughly how much
conformer data and wall-clock time that implies.

    python -m pdbenergy.cli inventory --no-table   # refresh the report first
    python scripts/estimate_workload.py

The time estimate uses the measured per-entry cost on the development machine
(2-4 minutes for 300-1650 prepared atoms with 8 OpenMM threads) and scales
linearly in prepared atom count.  It is an estimate, not a promise - time one
batch before committing to a long run.
"""

from __future__ import annotations

import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pdbenergy.config import Config                                    # noqa: E402

INVENTORY = os.path.join(ROOT, "outputs", "inventory.json")

#: Prepared (all-atom, hydrogenated) atoms are roughly 2x the heavy-atom count,
#: which is itself ~8 per residue.  Calibrated on the shipped 14-protein set:
#: 1CRN is 46 residues -> 642 prepared atoms (13.9/residue).
PREPARED_ATOMS_PER_RESIDUE = 13.9

#: Measured: ~2 min for a 460-atom protein, ~4 min for a 1650-atom one, with the
#: shipped sampling settings and 8 OpenMM threads.  Fitted as a + b*atoms.
MINUTES_PER_ENTRY_AT_460 = 2.0
MINUTES_PER_ENTRY_AT_1650 = 4.0


def minutes_per_entry(prepared_atoms: float) -> float:
    span = MINUTES_PER_ENTRY_AT_1650 - MINUTES_PER_ENTRY_AT_460
    slope = span / (1650 - 460)
    return max(0.5, MINUTES_PER_ENTRY_AT_460 + slope * (prepared_atoms - 460))


def usable_at_cap(report: dict, cap: int, min_residues: int) -> bool:
    """Re-evaluate usability for an arbitrary cap.

    ``inventory.json`` stores ``is_usable`` computed at whatever cap was used for
    that run, so it cannot be reused for a different cap.  The ``reason`` field
    still tells us *why* an entry failed, which is what we need: an entry rejected
    only for exceeding the cap becomes usable when the cap is raised, while one
    rejected for having no amino acids never will.
    """
    reason = report.get("reason", "") or ""
    if reason.startswith("parse error"):
        return False
    if "no standard amino-acid" in reason or "multi-chain" in reason:
        return False
    n_res = report.get("n_residues", 0)
    return min_residues <= n_res <= cap


def main() -> int:
    if not os.path.exists(INVENTORY):
        print(f"{INVENTORY} not found - run:  python -m pdbenergy.cli inventory --no-table")
        return 1

    with io.open(INVENTORY, encoding="utf-8") as fh:
        reports = json.load(fh)

    cfg = Config()
    ecfg = cfg.ensemble
    frames_per_entry = (
        len(ecfg.torsion_levels) * ecfg.n_torsion_per_level
        + (len(ecfg.md_temperatures) if ecfg.md_ps_per_temperature > 0 else 0)
        * max(1, int(round(ecfg.md_ps_per_temperature / ecfg.md_save_every_ps)))
        + ecfg.max_nmr_models
    )

    print(f"inventory: {len(reports)} entries")
    print(f"sampling: about {frames_per_entry} conformers per entry with the current config")
    print(f"          ({len(ecfg.torsion_levels)} torsion levels x "
          f"{ecfg.n_torsion_per_level}, MD at {ecfg.md_temperatures}, "
          f"up to {ecfg.max_nmr_models} NMR models)\n")

    header = (f"{'max_res':>8} {'entries':>8} {'conformers':>11} {'npz on disk':>12} "
              f"{'1 process':>11} {'4 workers':>11}")
    print(header)
    print("-" * len(header))
    print("(conformers are capped at 4e6 to keep the projection readable)\n")

    for cap in (120, 200, 300, 500):
        usable = [r for r in reports if usable_at_cap(r, cap, cfg.prepare.min_residues)]
        if not usable:
            continue
        conformers = len(usable) * frames_per_entry
        # Measured on the shipped set: 3.78 MB on disk for 730 conformers.
        disk_mb = conformers * (3.78 / 730)
        total_min = sum(
            minutes_per_entry(r["n_residues"] * PREPARED_ATOMS_PER_RESIDUE)
            for r in usable
        )
        # Measured: 4 workers x 2 threads on 4 physical cores gives about 2.6x the
        # throughput of one 8-thread process (OpenMM CPU scaling is sublinear).
        parallel_min = total_min / 2.6
        flag = "  <- over budget" if conformers > 4_000_000 else ""
        print(f"{cap:>8} {len(usable):>8} {conformers:>11,} {disk_mb:>9.0f} MB "
              f"{total_min / 60:>9.1f} h {parallel_min / 60:>9.1f} h{flag}")

    print("\nNotes")
    print("  * 'npz on disk' is data/interim/<ID>.npz, committed to git - keep it modest")
    print("  * the graph cache holds ~0.5 MB per conformer IN RAM: 20k conformers ~ 10 GB")
    print("    (the iterate driver disables the cache automatically past --cache-budget)")
    print("  * labelling is resumable: finished entries are skipped on the next run")
    print("  * estimate only - time one batch of 20 before committing to a long run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
