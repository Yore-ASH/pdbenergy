"""Cache featurised graphs on disk.

Why this exists
---------------
Featurising a conformation costs ~150 ms: an O(N^2) distance matrix, a
neighbour selection, and (before the refactor) an edge-feature array.  Training
touches every sample once per epoch, so 1300 samples x 200 epochs would spend
**11 hours** just rebuilding identical graphs that never change.

The graphs depend only on (coordinates, feature config), never on the model, the
epoch, or the target.  So they are computed once, stored, and reused.  Because
edge features are just ``[distance, bond_flag]`` the cache is small enough to
hold in RAM (~0.4 GB for this dataset) and loading it takes seconds.

The cache key includes a hash of the feature configuration, so changing the
cutoff or basis invalidates it automatically instead of silently serving stale
graphs with the wrong number of edge channels.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from typing import Sequence

import numpy as np
import torch

from .config import FeatureConfig
from .features import GraphArrays, frame_to_graph


def feature_signature(feature_cfg: FeatureConfig, extra: dict | None = None) -> str:
    """Stable short hash of everything that changes the graph layout."""
    payload = asdict(feature_cfg)
    if extra:
        payload.update(extra)
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def cache_file(cache_dir: str, feature_cfg: FeatureConfig, extra: dict | None = None) -> str:
    return os.path.join(cache_dir, f"graphs_{feature_signature(feature_cfg, extra)}.pt")


def build_cache(
    ensembles: dict,
    rows: Sequence[tuple[str, int]],
    feature_cfg: FeatureConfig,
    *,
    verbose: bool = True,
) -> dict[tuple[str, int], GraphArrays]:
    """Featurise every requested (protein, frame) pair."""
    cache: dict[tuple[str, int], GraphArrays] = {}
    total = len(rows)
    for i, (protein, frame) in enumerate(rows, start=1):
        ensemble = ensembles[protein]
        key = (protein, int(frame))
        if key in cache:
            continue
        cache[key] = frame_to_graph(
            np.asarray(ensemble.coords[frame], dtype=np.float64),
            [str(e) for e in ensemble.elements],
            feature_cfg,
        )
        if verbose and (i % 200 == 0 or i == total):
            print(f"    featurised {i}/{total}", flush=True)
    return cache


def load_or_build(
    ensembles: dict,
    rows: Sequence[tuple[str, int]],
    feature_cfg: FeatureConfig,
    cache_dir: str,
    *,
    enabled: bool = True,
    verbose: bool = True,
) -> dict[tuple[str, int], GraphArrays] | None:
    """Return a graph cache, building and saving it on first use.

    Returns ``None`` when caching is disabled, in which case the dataset
    featurises on the fly (slower but uses no disk/memory).
    """
    if not enabled:
        return None

    extra = {"n_proteins": len(ensembles), "n_rows": len(rows)}
    path = cache_file(cache_dir, feature_cfg, extra)
    if os.path.exists(path):
        try:
            cache = torch.load(path, map_location="cpu", weights_only=False)
            if verbose:
                print(f"    loaded graph cache: {path} ({len(cache)} graphs)", flush=True)
            missing = [r for r in rows if (r[0], int(r[1])) not in cache]
            if not missing:
                return cache
            if verbose:
                print(f"    cache incomplete ({len(missing)} missing), extending", flush=True)
            cache.update(build_cache(ensembles, missing, feature_cfg, verbose=False))
        except Exception as exc:  # corrupt cache must never be fatal
            if verbose:
                print(f"    cache unreadable ({type(exc).__name__}), rebuilding", flush=True)
            cache = build_cache(ensembles, rows, feature_cfg, verbose=verbose)
    else:
        if verbose:
            print(f"    building graph cache for {len(rows)} frames...", flush=True)
        cache = build_cache(ensembles, rows, feature_cfg, verbose=verbose)

    os.makedirs(cache_dir, exist_ok=True)
    torch.save(cache, path)
    if verbose:
        size_mb = os.path.getsize(path) / 1e6
        print(f"    saved graph cache: {path} ({size_mb:.0f} MB)", flush=True)
    return cache
