# Examples

Three ways to drive the project, from thinnest to thickest.

## 1. Command line (normal use)

```powershell
# one-time setup
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# fetch structures -> label them with physics -> train -> evaluate
python -m pdbenergy.cli download
python -m pdbenergy.cli ensemble --threads 8
python -m pdbenergy.cli dataset
python -m pdbenergy.cli train --model schnet
python -m pdbenergy.cli evaluate --run-dir outputs/schnet_protein

# score your own file
python -m pdbenergy.cli predict my_structure.pdb --verify
```

After `pip install -e .` the same CLI is available as a plain command:

```powershell
pdbenergy --help
pdbenergy predict my_structure.pdb --verify
```

Everything in one go, with per-stage logs under `logs/`:

```powershell
.\scripts\run_pipeline.ps1                 # full run
.\scripts\run_pipeline.ps1 -SkipEnsemble   # reuse the ensembles already on disk
```

## 2. Small scripts (this folder)

| script | what it shows |
|---|---|
| `predict_pdb.py` | load a checkpoint once, score one or many PDB files, rank them, optionally check against the true force-field energy |
| `train_from_python.py` | drive the library directly: `Config` → `load_ensembles` → `build_bundle` → `train_model` → `evaluate_split` |

```powershell
python examples\predict_pdb.py data\raw\1L2Y.pdb --models 3 --verify
python examples\train_from_python.py --epochs 5 --out-dir outputs/demo
```

## 3. The library itself

```python
from pdbenergy.config import Config
from pdbenergy.dataset import build_bundle, load_ensembles
from pdbenergy.train import train_model

cfg = Config()
cfg.model.hidden_dim = 128          # any knob is a dataclass field
cfg.features.cutoff = 4.5
cfg.train.epochs = 100

ensembles = load_ensembles("data/interim")
bundle = build_bundle(ensembles, cfg)     # protein-level split, no leakage
result = train_model(bundle, out_dir="outputs/my_run")
```

```python
from pdbenergy.predict import EnergyPredictor

predictor = EnergyPredictor("outputs/schnet_protein/checkpoint.pt")
for pdb_file in ["a.pdb", "b.pdb", "c.pdb"]:
    for pred in predictor.predict_file(pdb_file, max_models=1):
        print(pdb_file, pred.predicted_relative_energy, "kcal/mol")
```

## What the numbers mean

* The model predicts **relative** conformational energy `ΔE = E − min(E)` for a
  protein, in kcal/mol — how much worse this conformer is than that protein's
  best-sampled structure. It is meaningful for comparing conformers of the same
  molecule, not for comparing two different proteins.
* A raw PDB file almost never contains hydrogens, but the force field (and
  therefore the training labels) is all-atom, so `predict` first adds hydrogens
  with PDBFixer at pH 7. This is fast; the expensive step the network replaces is
  the energy evaluation itself.
* The shipped model is a demonstration, not a production scoring function: see
  the honest results table in the top-level `README.md` and chapter 11 of
  `TeachFlow.md`.
