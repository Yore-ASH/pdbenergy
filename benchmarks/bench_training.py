"""Measure training step time for realistic graph batches and model sizes."""
import sys, time, os
sys.path.insert(0, ".")
import torch
import numpy as np

from pdbenergy.config import ModelConfig, FeatureConfig
from pdbenergy.features import frame_to_graph, collate_graphs
from pdbenergy.models import build_model, edge_dims_for

torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "1")))

# A realistic protein-like graph: ~800 atoms in a globule.
rng = np.random.default_rng(0)
n_atoms = 800
coords = rng.normal(scale=9.0, size=(n_atoms, 3))          # radius ~15 A globule
elements = rng.choice(["C", "N", "O", "H", "S"], size=n_atoms,
                      p=[0.32, 0.09, 0.10, 0.48, 0.01])

configs = [
    ("hidden128 int4 cut4.5 rbf48 nbr40", dict(hidden_dim=128, n_interactions=4),
     FeatureConfig(cutoff=4.5, n_rbf=48, max_neighbors=40)),
    ("hidden96  int4 cut4.0 rbf40 nbr36", dict(hidden_dim=96, n_interactions=4),
     FeatureConfig(cutoff=4.0, n_rbf=40, max_neighbors=36)),
    ("hidden64  int3 cut3.5 rbf32 nbr32", dict(hidden_dim=64, n_interactions=3),
     FeatureConfig(cutoff=3.5, n_rbf=32, max_neighbors=32)),
    ("hidden64  int4 cut4.0 rbf32 nbr32", dict(hidden_dim=64, n_interactions=4),
     FeatureConfig(cutoff=4.0, n_rbf=32, max_neighbors=32)),
]

for batch_size in (8, 16):
    print(f"===== batch_size={batch_size}  threads={torch.get_num_threads()} =====")
    for name, mkw, fcfg in configs:
        g = frame_to_graph(coords, elements, fcfg)
        items = []
        for i in range(batch_size):
            items.append({
                "z": torch.from_numpy(g.z), "pos": torch.from_numpy(g.pos),
                "edge_index": torch.from_numpy(g.edge_index),
                "edge_attr": torch.from_numpy(g.edge_attr),
                "y": torch.tensor(float(i)), "protein": "X", "frame": i, "source": "b",
            })
        batch = collate_graphs(items)
        n_edges = batch["edge_index"].shape[1]

        model = build_model(ModelConfig(kind="schnet", **mkw), **edge_dims_for(fcfg))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        lossf = torch.nn.HuberLoss(delta=1.0)

        def step():
            opt.zero_grad(set_to_none=True)
            out = model(z=batch["z"], edge_index=batch["edge_index"],
                        edge_attr=batch["edge_attr"], batch=batch["batch"],
                        n_graphs=batch["n_graphs"])
            loss = lossf(out, batch["y"])
            loss.backward()
            opt.step()

        step()  # warm up
        t0 = time.time()
        n = 3
        for _ in range(n):
            step()
        dt = (time.time() - t0) / n
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name:<36} edges={n_edges:>8,} params={n_params:>8,} "
              f"step={dt*1000:8.0f} ms")
