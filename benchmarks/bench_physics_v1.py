"""Benchmark: how much does each physics setting cost?"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
from pdbenergy.pdbio import read_pdb, protein_only
from pdbenergy.config import LabelConfig, PrepareConfig
from pdbenergy.labels import OpenMMEnergyEngine

models = read_pdb("data/raw/1CRN.pdb")
struct = protein_only(models[0])

configs = {
    "GB + NoCutoff": LabelConfig(forcefield_files=("amber14-all.xml", "implicit/gbn2.xml"),
                                 implicit_solvent="OBC2", nonbonded_method="NoCutoff"),
    "GB + cutoff1.2": LabelConfig(forcefield_files=("amber14-all.xml", "implicit/gbn2.xml"),
                                  implicit_solvent="OBC2", nonbonded_method="CutoffNonPeriodic",
                                  cutoff_nm=1.2),
    "vacuum + NoCutoff": LabelConfig(forcefield_files=("amber14-all.xml",),
                                     implicit_solvent=None, nonbonded_method="NoCutoff"),
    "vacuum + cutoff1.2": LabelConfig(forcefield_files=("amber14-all.xml",),
                                      implicit_solvent=None, nonbonded_method="CutoffNonPeriodic",
                                      cutoff_nm=1.2),
}

for threads in (1, 8):
    print(f"================ Threads={threads} ================")
    for name, cfg in configs.items():
        try:
            t0 = time.time()
            eng = OpenMMEnergyEngine(cfg, PrepareConfig(), threads=threads)
            prep = eng.prepare(struct, f"1CRN_t{threads}", workdir="data/interim/bench")
            t_prep = time.time() - t0

            ctx, _ = eng.make_context(prep)
            eng.set_positions(ctx, prep.positions_nm * 10.0)
            eng.energy_kcal(ctx)  # warm up
            t0 = time.time()
            n = 10
            for _ in range(n):
                e = eng.energy_kcal(ctx)
            t_energy = (time.time() - t0) / n

            t0 = time.time()
            eng.run_md(prep, temperature_k=300.0, n_steps=200, save_every=100, seed=1)
            t_md = (time.time() - t0) / 200 * 1000  # ms per step

            ctx2, _ = eng.make_context(prep)
            eng.set_positions(ctx2, prep.positions_nm * 10.0)
            t0 = time.time()
            eng.minimise(ctx2, 50)
            t_min50 = time.time() - t0

            print(f"{name:<20} prep {t_prep:5.1f}s | E {t_energy*1000:6.1f} ms | "
                  f"MD {t_md:6.2f} ms/step | min50 {t_min50:6.2f}s | E0={e:.1f}")
        except Exception as exc:
            print(f"{name:<20} FAILED: {type(exc).__name__}: {exc}")
