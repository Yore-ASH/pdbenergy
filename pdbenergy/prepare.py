"""Downloading and inventorying raw PDB entries.

Two jobs live here:

1. :func:`download_pdb` - fetch ``<ID>.pdb`` from the RCSB Protein Data Bank.
2. :func:`inventory` - scan a directory of PDB files and report what is inside,
   so that the pipeline never silently trains on something unexpected
   (DNA, a huge complex, or a file with no protein at all).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from typing import Iterable

from .config import DEFAULT_PDB_IDS
from .pdbio import (
    BACKBONE_ATOMS,
    STANDARD_AA,
    Structure,
    chain_summary,
    protein_only,
    read_pdb,
)

#: RCSB serves the legacy PDB-format file at this URL.  ``.cif`` is available at
#: the same path with a different extension.
RCSB_DOWNLOAD = "https://files.rcsb.org/download/{pdb_id}.pdb"


def download_pdb(
    pdb_id: str,
    out_dir: str,
    *,
    force: bool = False,
    retries: int = 3,
    timeout: int = 60,
) -> str:
    """Download one entry, skipping the network when the file already exists.

    Large NMR ensembles exceed 1 MB, so the raw files are *not* deleted after
    preparation: keeping them makes the dataset re-derivable offline.
    """
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, f"{pdb_id.upper()}.pdb")
    if os.path.exists(target) and not force and os.path.getsize(target) > 0:
        return target

    url = RCSB_DOWNLOAD.format(pdb_id=pdb_id.upper())
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = response.read()
            if not payload:
                raise ValueError("empty response")
            with open(target, "wb") as fh:
                fh.write(payload)
            return target
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"could not download {pdb_id} from {url}: {last_error}")


def download_many(pdb_ids: Iterable[str], out_dir: str, *, force: bool = False, verbose: bool = True) -> dict[str, str]:
    """Download several entries; failures are reported but do not abort the run."""
    paths: dict[str, str] = {}
    for pdb_id in pdb_ids:
        pdb_id = pdb_id.strip().upper()
        if not pdb_id:
            continue
        try:
            path = download_pdb(pdb_id, out_dir, force=force)
            paths[pdb_id] = path
            if verbose:
                print(f"  ok   {pdb_id}  {os.path.getsize(path):>9,d} bytes", flush=True)
        except Exception as exc:
            print(f"  FAIL {pdb_id}  {exc}", flush=True)
    return paths


def discover_pdb_files(raw_dir: str) -> dict[str, str]:
    """Map PDB ID -> file path for every ``*.pdb`` / ``*.ent`` / ``*.pdb.gz`` file."""
    if not os.path.isdir(raw_dir):
        return {}
    found: dict[str, str] = {}
    for name in sorted(os.listdir(raw_dir)):
        lower = name.lower()
        if not lower.endswith((".pdb", ".ent", ".pdb.gz", ".ent.gz")):
            continue
        pdb_id = name.split(".")[0].upper()
        found[pdb_id] = os.path.join(raw_dir, name)
    return found


@dataclass
class EntryReport:
    """What one raw file actually contains."""

    pdb_id: str
    path: str
    file_size: int
    n_models: int
    n_atoms: int
    n_protein_atoms: int
    n_residues: int
    n_chains: int
    sequence: str
    experiment: str
    hetatm_residues: list[str]
    is_usable: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def inventory(raw_dir: str, *, max_residues: int = 120) -> list[EntryReport]:
    """Summarise every PDB file in ``raw_dir`` and flag the unusable ones."""
    reports: list[EntryReport] = []
    for pdb_id, path in discover_pdb_files(raw_dir).items():
        try:
            models = read_pdb(path)
        except Exception as exc:
            reports.append(
                EntryReport(pdb_id, path, os.path.getsize(path), 0, 0, 0, 0, 0, "", "", [], False,
                            f"parse error: {exc}")
            )
            continue
        first = models[0]
        clean = protein_only(first, keep_hydrogens=True)
        residue_names = sorted({a.resname for a in first.atoms if a.record == "HETATM"})
        n_res = clean.n_residues()
        chains = {a.chain for a in clean.atoms}
        usable, reason = True, ""
        if n_res == 0:
            usable, reason = False, "no standard amino-acid residues"
        elif n_res > max_residues:
            usable, reason = False, f"{n_res} residues exceeds max_residues={max_residues}"
        elif len(chains) > 4:
            usable, reason = False, f"{len(chains)} chains (multi-chain assemblies unsupported)"

        experiment = ""
        try:
            with open(path, "r", errors="replace") as fh:
                for line in fh:
                    if line.startswith("EXPDTA"):
                        experiment = line[10:].strip()
                        break
        except Exception:
            pass

        reports.append(
            EntryReport(
                pdb_id=pdb_id,
                path=path,
                file_size=os.path.getsize(path),
                n_models=len(models),
                n_atoms=len(first),
                n_protein_atoms=len(clean),
                n_residues=n_res,
                n_chains=len(chains),
                sequence="".join(protein_only(first).sequence().values()),
                experiment=experiment,
                hetatm_residues=residue_names,
                is_usable=usable,
                reason=reason,
            )
        )
    return reports


def write_json(path: str, payload) -> str:
    """Small helper: pretty-print JSON next to the data it describes."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def print_inventory(reports: Iterable[EntryReport]) -> None:
    """Tabular console report."""
    print(f"{'ID':<6} {'models':>6} {'atoms':>7} {'prot':>7} {'res':>5} {'ch':>3} "
          f"{'usable':>6}  experiment")
    print("-" * 92)
    for r in reports:
        print(
            f"{r.pdb_id:<6} {r.n_models:>6} {r.n_atoms:>7} {r.n_protein_atoms:>7} "
            f"{r.n_residues:>5} {r.n_chains:>3} {'yes' if r.is_usable else 'NO':>6}  "
            f"{r.experiment[:40]}"
            + (f"  <- {r.reason}" if r.reason else "")
        )


__all__ = [
    "DEFAULT_PDB_IDS",
    "RCSB_DOWNLOAD",
    "STANDARD_AA",
    "BACKBONE_ATOMS",
    "chain_summary",
    "discover_pdb_files",
    "download_many",
    "download_pdb",
    "inventory",
    "print_inventory",
    "write_json",
]
