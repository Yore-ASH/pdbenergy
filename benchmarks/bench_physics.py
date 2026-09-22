"""Benchmark 2: GB with a real cutoff vs NoCutoff, and vacuum reference."""
import sys, time
sys.path.insert(0, ".")
from pdbenergy.pdbio import read_pdb, protein_only
from pdbenergy.config import LabelConfig, PrepareConfig
from pdbenergy.labels import OpenMMEnergyEngine

models = read_pdb("data/raw/1CRN.pdb")
struct = protein_only(models[0])

base = dict(forcefield_files=("amber14-all.xml", "implicit/gbn2.xml"), implicit_solvent="OBC2")
configs = {
    "GB nocutoff": LabelConfig(nonbonded_method="NoCutoff", **base),
    "GB cut 1.0nm": LabelConfig(nonbonded_method="CutoffNonPeriodic", cutoff_nm=1.0, **base),
    "GB cut 1.2nm": LabelConfig(nonbonded_method="CutoffNonPeriodic", cutoff_nm=1.2, **base),
    "vacuum cut1.2": LabelConfig(forcefield_files=("amber14-all.xml",), implicit_solvent=None,
                                 nonbonded_method="CutoffNonPeriodic", cutoff_nm=1.2),
}

for threads in (8,):
    print(f"================ Threads={threads}  (1CRN, 642 atoms) ================")
    for name, cfg in configs.items():
        try:
            t0 = time.time()
            eng = OpenMMEnergyEngine(cfg, PrepareConfig(), threads=threads)
            prep = eng.prepare(struct, "1CRN", workdir="data/interim/bench")
            t_prep = time.time() - t0
            ctx, _ = eng.make_context(prep)
            eng.set_positions(ctx, prep.positions_nm * 10.0)
            e = eng.energy_kcal(ctx)
            t0 = time.time()
            for _ in range(10):
                e = eng.energy_kcal(ctx)
            t_energy = (time.time() - t0) / 10
            t0 = time.time()
            eng.run_md(prep, temperature_k=300.0, n_steps=200, save_every=100, seed=1)
            t_md = (time.time() - t0) / 200 * 1000
            ctx2, _ = eng.make_context(prep)
            eng.set_positions(ctx2, prep.positions_nm * 10.0)
            t0 = time.time()
            e_min = eng.minimise(ctx2, 50)
            t_min50 = time.time() - t0
            print(f"{name:<14} prep {t_prep:5.1f}s | E {t_energy*1000:7.2f} ms | "
                  f"MD {t_md:7.2f} ms/step | min50 {t_min50:7.2f}s | E={e:9.1f} Emin={e_min:9.1f}")
        except Exception as exc:
            print(f"{name:<14} FAILED: {type(exc).__name__}: {exc}")
