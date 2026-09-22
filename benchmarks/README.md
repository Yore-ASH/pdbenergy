# benchmarks

Small scripts that answer "how expensive is this, really?" before committing to a
long run. Every number quoted in [`../TeachFlow.md`](../TeachFlow.md) comes from
one of these.

Run them from the repository root with the project venv, e.g.

```powershell
.\.venv\Scripts\python.exe benchmarks\bench_physics.py
```

| script | question it answers |
|---|---|
| `bench_physics_v1.py` | Which physics setup can we afford: implicit solvent vs vacuum, and how much does OpenMM threading help? |
| `bench_physics.py` | How much does a finite non-bonded cutoff save over an exact `NoCutoff` treatment? |
| `bench_training.py` | What do model width, interaction count, cutoff and basis size cost per training step? |
| `bench_ops.py` | Where inside a single training step does the time actually go? |
| `validate_ensemble.py` | Does one protein's ensemble look physically sane (energy range, bond terms, source mix)? |

## Measured results on the development machine

Intel Core i7-8550U (4 cores / 8 threads), 24 GB RAM, CPU-only PyTorch 2.14.
Absolute numbers will differ on your machine; the *ratios* are the durable part.

### Physics: 1CRN, 642 atoms, AMBER14

| setting | single-point energy | MD step | 50 L-BFGS iterations |
|---|---|---|---|
| GBn2, `NoCutoff` | 65.6 ms | 80.5 ms | 23.8 s |
| GBn2, `CutoffNonPeriodic` 1.0 nm | **22.0 ms** | **22.2 ms** | **6.1 s** |
| GBn2, `CutoffNonPeriodic` 1.2 nm | 35.9 ms | 34.4 ms | 6.3 s |
| vacuum, `CutoffNonPeriodic` 1.2 nm | 1.1 ms | 1.5 ms | 0.35 s |

Conclusions that shaped the project:

* implicit solvent with a 1.0 nm cutoff is **3x cheaper** than the exact treatment
  and is standard practice in GB simulations, so it is the default;
* vacuum would be another ~20x cheaper, but drops solvent screening entirely.
  It is kept as a documented option (`label.implicit_solvent = null`), not the default;
* threading gives ~3.5-4x at 8 threads, well short of linear scaling.

### Model sizing: batch of 8x 800-atom graphs, 4 threads (under load)

| model | edges | params | step time |
|---|---|---|---|
| hidden 128, 4 interactions, cutoff 4.5 A, 48 RBF | 54,592 | 185,986 | 17.0 s |
| hidden 96, 4 interactions, cutoff 4.0 A, 40 RBF | 38,224 | 109,538 | 12.4 s |
| hidden 64, 4 interactions, cutoff 4.0 A, 32 RBF | 38,224 | 53,058 | 12.3 s |
| hidden 64, 3 interactions, cutoff 3.5 A, 32 RBF | 25,616 | 42,562 | 9.1 s |

The default was moved from the first row to roughly the third/fourth row: a ~12x
difference in wall-clock that decides whether an experiment finishes at all.

### Where a training step goes (one 800-atom graph, hidden 128)

| component | time |
|---|---|
| Gaussian RBF expansion (48 basis, all edges) | 788 ms |
| filter MLP (49 -> 128 -> 128) | 441 ms |
| `x[sender]` gather | 76 ms |
| gather x weight + scatter-add | 639 ms |
| one interaction block (forward) | 1194 ms |
| full forward (4 blocks) | 6467 ms |
| full forward + backward + optimizer step | 15,620 ms |

Message passing is dominated by the per-edge filter MLP, which is why edge count
(and therefore the cutoff) is the single most effective knob.
