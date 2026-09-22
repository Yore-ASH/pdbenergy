"""Validate the default ensemble build on one protein, with timings."""
import sys, time
sys.path.insert(0, ".")
import numpy as np
from pdbenergy.config import Config
from pdbenergy.ensemble import build_ensemble, build_one
from pdbenergy.labels import OpenMMEnergyEngine

cfg = Config()
t0 = time.time()
eng = OpenMMEnergyEngine(cfg.label, cfg.prepare, threads=8, verbose=True)
pdb_id, target, summary = build_one(
    "1CRN", "data/raw/1CRN.pdb", "data/interim",
    engine=eng, ensemble_cfg=cfg.ensemble, overwrite=True, verbose=True,
)
print(f"\nelapsed {time.time()-t0:.1f}s")
print("summary:", summary)
ens = __import__("pdbenergy.ensemble", fromlist=["ProteinEnsemble"]).ProteinEnsemble.load(target)
print("coords", ens.coords.shape, "elements", sorted(set(ens.elements.tolist())))
print("sources:", {s: int((ens.source == s).sum()) for s in sorted(set(ens.source))})
print("dE percentiles:", np.percentile(ens.relative_energy, [0, 25, 50, 75, 90, 99, 100]).round(2))
print("terms keys:", list(ens.energy_terms.keys()))
print("bond term range:", float(ens.energy_terms['bond'].min()), float(ens.energy_terms['bond'].max()))
