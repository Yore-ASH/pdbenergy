"""pdbenergy - learn a neural network surrogate for protein conformational energy.

The package implements a complete, reproducible pipeline:

    PDB files  ->  clean & prepare  ->  conformational ensembles  ->  physics energy labels
               ->  graph featurisation  ->  PyTorch message-passing network  ->  evaluation

See ``TeachFlow.md`` in the repository root for the full teaching notes.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
