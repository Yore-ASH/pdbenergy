"""Profile the components of one training step and check BLAS throughput."""
import os, sys, time
sys.path.insert(0, ".")
import numpy as np
import torch

threads = int(os.environ.get("BENCH_THREADS", "1"))
torch.set_num_threads(threads)
print("torch threads:", torch.get_num_threads(), "| interop:", torch.get_num_interop_threads())

# --- 1. raw BLAS throughput ------------------------------------------------
for shape in [(512, 512, 512), (54592, 49, 128), (54592, 128, 128)]:
    m, k, n = shape
    a = torch.randn(m, k)
    b = torch.randn(k, n)
    for _ in range(2):
        c = a @ b
    t0 = time.time()
    reps = 5
    for _ in range(reps):
        c = a @ b
    dt = (time.time() - t0) / reps
    gflops = 2 * m * k * n / dt / 1e9
    print(f"  GEMM {m}x{k}@{k}x{n}: {dt*1000:8.2f} ms  {gflops:7.2f} GFLOPS")

# --- 2. component timings --------------------------------------------------
from pdbenergy.config import FeatureConfig, ModelConfig
from pdbenergy.features import frame_to_graph, collate_graphs
from pdbenergy.models import GaussianRBFExpansion, InteractionBlock, build_model, edge_dims_for, scatter_sum

fcfg = FeatureConfig()
mcfg = ModelConfig(kind="schnet", hidden_dim=128, n_interactions=4)
n_atoms = 800
rng = np.random.default_rng(0)
coords = rng.normal(scale=9.0, size=(n_atoms, 3))
elements = rng.choice(["C", "N", "O", "H", "S"], size=n_atoms, p=[.32, .09, .10, .48, .01])
g = frame_to_graph(coords, elements, fcfg)
items = [{"z": torch.from_numpy(g.z), "pos": torch.from_numpy(g.pos),
          "edge_index": torch.from_numpy(g.edge_index),
          "edge_attr": torch.from_numpy(g.edge_attr),
          "y": torch.tensor(0.0), "protein": "X", "frame": i, "source": "b"} for i in range(8)]
batch = collate_graphs(items)
E = batch["edge_index"].shape[1]
N = batch["z"].shape[0]
print(f"\n  batch: atoms={N:,} edges={E:,}")

def timeit(fn, reps=3, label=""):
    fn()
    t0 = time.time()
    for _ in range(reps):
        fn()
    dt = (time.time() - t0) / reps
    print(f"  {label:<42} {dt*1000:9.2f} ms")
    return dt

rbf = GaussianRBFExpansion(48, 4.5, 0.0, 5.0)
timeit(lambda: rbf(batch["edge_attr"][:, 0]), label="RBF expansion (48 basis, all edges)")

x = torch.randn(N, 128)
block = InteractionBlock(128, 48, 1)
expanded = torch.cat([rbf(batch["edge_attr"][:, 0]), batch["edge_attr"][:, 1:]], dim=-1)
timeit(lambda: block.filter(expanded), label="  just filter MLP")
timeit(lambda: x[batch["edge_index"][1]], label="  gather x[sender]")
w = block.filter(expanded)
timeit(lambda: scatter_sum(x[batch["edge_index"][1]] * w, batch["edge_index"][0], N),
       label="  gather*weight + scatter")
timeit(lambda: block(x, batch["edge_index"], expanded, N), label="one InteractionBlock (fwd)")

model = build_model(mcfg, **edge_dims_for(fcfg))
timeit(lambda: model(z=batch["z"], edge_index=batch["edge_index"], edge_attr=batch["edge_attr"],
                    batch=batch["batch"], n_graphs=batch["n_graphs"]), label="full model forward")

opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
lossf = torch.nn.HuberLoss()
def full_step():
    opt.zero_grad(set_to_none=True)
    out = model(z=batch["z"], edge_index=batch["edge_index"], edge_attr=batch["edge_attr"],
                batch=batch["batch"], n_graphs=batch["n_graphs"])
    lossf(out, batch["y"]).backward()
    opt.step()
timeit(full_step, reps=2, label="full fwd+bwd+step")
