"""Predict conformational energy for arbitrary PDB files.

This is the user-facing half of the project: point it at a PDB file and it
returns an energy in kcal/mol, without running any physics.

The pipeline at inference time
------------------------------
1. parse the PDB, keep standard amino acids, resolve altlocs;
2. protonate with PDBFixer (a raw PDB almost never contains hydrogens, and the
   force field - hence the labels - is an all-atom energy);
3. featurise into the same graph representation used for training;
4. run the network;
5. optionally evaluate the *true* force-field energy for the same structure so
   the prediction can be checked on the spot (``--verify``).

Interpretation
--------------
The network is trained on **relative conformational energy**: ``E - min(E)``
within a protein.  That is the quantity that is meaningful for comparing
conformations of the same molecule.  Absolute energies are dominated by
composition and long-range solvation and are reported only when a reference is
supplied (``--reference-energy``, or ``--verify`` which measures it).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .config import LabelConfig, PrepareConfig
from .features import frame_to_graph
from .labels import OpenMMEnergyEngine
from .pdbio import Structure, protein_only, read_pdb
from .train import load_checkpoint


@dataclass
class Prediction:
    """One predicted (and optionally verified) structure."""

    name: str
    model_index: int
    n_atoms: int
    n_residues: int
    predicted_relative_energy: float
    true_relative_energy: float | None = None
    error: float | None = None
    sequence: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model_index": self.model_index,
            "n_atoms": self.n_atoms,
            "n_residues": self.n_residues,
            "predicted_relative_energy_kcal_per_mol": self.predicted_relative_energy,
            "true_relative_energy_kcal_per_mol": self.true_relative_energy,
            "error_kcal_per_mol": self.error,
            "sequence": self.sequence,
        }


class EnergyPredictor:
    """Loads a checkpoint once and predicts energies for many structures."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cpu",
        label_cfg: LabelConfig | None = None,
        prepare_cfg: PrepareConfig | None = None,
        threads: int = 4,
    ):
        self.checkpoint_path = checkpoint_path
        payload = load_checkpoint(checkpoint_path, device=device)
        self.payload = payload
        self.model = payload["model"]
        self.feature_cfg = payload["_feature_config"]
        self.device = torch.device(device)
        self.norm_mean = float(payload["norm_mean"])
        self.norm_std = float(payload["norm_std"])
        self.descriptor_mode = bool(payload.get("descriptor_mode", False))
        if self.descriptor_mode:
            raise ValueError(
                "the descriptor baseline cannot be used for prediction from a PDB "
                "file alone; it exists only as a comparison baseline"
            )
        self.label_cfg = label_cfg or LabelConfig()
        self.prepare_cfg = prepare_cfg or PrepareConfig()
        self._engine: OpenMMEnergyEngine | None = None
        self._threads = threads

    # -- lazy helpers -------------------------------------------------------- #
    @property
    def engine(self) -> OpenMMEnergyEngine:
        if self._engine is None:
            self._engine = OpenMMEnergyEngine(self.label_cfg, self.prepare_cfg, threads=self._threads)
        return self._engine

    def _protonate(self, structure: Structure, name: str, workdir: str | None):
        """Add hydrogens and return (coords_angstrom, elements)."""
        topology, positions_nm = self.engine.protonate(structure, name, workdir=workdir)
        elements = [
            (a.element.symbol.upper() if a.element is not None else "X")
            for a in topology.atoms()
        ]
        return np.asarray(positions_nm, dtype=np.float64) * 10.0, elements

    @torch.no_grad()
    def _predict_graph(self, coords: np.ndarray, elements: Sequence[str]) -> float:
        graph = frame_to_graph(coords, elements, self.feature_cfg)
        batch = {
            "z": torch.from_numpy(graph.z),
            "pos": torch.from_numpy(graph.pos),
            "edge_index": torch.from_numpy(graph.edge_index),
            "edge_attr": torch.from_numpy(graph.edge_attr),
            "batch": torch.zeros(graph.n_atoms, dtype=torch.long),
            "n_graphs": 1,
        }
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = self.model(
            z=batch["z"],
            edge_index=batch["edge_index"],
            edge_attr=batch["edge_attr"],
            batch=batch["batch"],
            n_graphs=1,
        )
        normalised = float(out.detach().cpu().reshape(-1)[0])
        return normalised * self.norm_std + self.norm_mean

    # -- single structure ---------------------------------------------------- #
    def predict_structure(
        self,
        structure: Structure,
        *,
        name: str = "structure",
        model_index: int = 1,
        workdir: str | None = None,
    ) -> tuple[Prediction, np.ndarray, list[str]]:
        """Predict for one parsed model; also returns its prepared coordinates."""
        coords, elements = self._protonate(structure, name, workdir)
        energy = self._predict_graph(coords, elements)
        pred = Prediction(
            name=name,
            model_index=model_index,
            n_atoms=len(elements),
            n_residues=structure.n_residues(),
            predicted_relative_energy=energy,
            sequence="".join(protein_only(structure).sequence().values()),
        )
        return pred, coords, elements

    # -- files --------------------------------------------------------------- #
    def predict_file(
        self,
        path: str,
        *,
        max_models: int = 1,
        verify: bool = False,
        workdir: str | None = None,
        verbose: bool = True,
    ) -> list[Prediction]:
        """Predict for every (up to ``max_models``) model in a PDB file.

        With ``verify=True`` the true AMBER14+GB energy is computed for the same
        prepared structures, so the error of the surrogate is measured rather
        than assumed.  Because the learned target is *relative* energy, the
        verified reference is the lowest true energy among the models processed.
        """
        name = os.path.splitext(os.path.basename(path))[0]
        models = read_pdb(path)
        if not models:
            raise ValueError(f"{path}: no atoms found")
        models = models[: max(1, max_models)]

        workdir = workdir or os.path.dirname(os.path.abspath(path))
        predictions: list[Prediction] = []
        prepared: list[tuple[np.ndarray, list[str]]] = []
        for i, structure in enumerate(models, start=1):
            pred, coords, elements = self.predict_structure(
                structure, name=name, model_index=i, workdir=workdir
            )
            predictions.append(pred)
            prepared.append((coords, elements))
            if verbose:
                print(
                    f"  {name} model {i}: {pred.n_residues} residues, "
                    f"{pred.n_atoms} atoms -> dE = {pred.predicted_relative_energy:8.3f} kcal/mol",
                    flush=True,
                )

        if verify:
            true_energies: list[float] = []
            for i, structure in enumerate(models, start=1):
                engine = self.engine
                prepared_sys = engine.prepare(structure, f"{name}_m{i}", workdir=workdir)
                energy, _ = engine.snapshot_energy_kcal(
                    prepared_sys, prepared[i - 1][0]
                )
                true_energies.append(energy)
            reference = min(true_energies)
            for pred, energy in zip(predictions, true_energies):
                pred.true_relative_energy = energy - reference
                pred.error = pred.predicted_relative_energy - pred.true_relative_energy
            if verbose:
                for pred in predictions:
                    print(
                        f"  {name} model {pred.model_index}: true dE = "
                        f"{pred.true_relative_energy:8.3f} | predicted = "
                        f"{pred.predicted_relative_energy:8.3f} | error = {pred.error:7.3f} kcal/mol",
                        flush=True,
                    )
        return predictions

    def rank_files(self, paths: Sequence[str], *, verify: bool = False) -> list[dict]:
        """Score several PDB files and return them sorted by predicted energy.

        This is the realistic use case for a learned energy: given a set of
        candidate conformations, put the most favourable ones first.
        """
        rows: list[dict] = []
        for path in paths:
            for pred in self.predict_file(path, max_models=1, verify=verify, verbose=False):
                row = pred.to_dict()
                row["path"] = path
                rows.append(row)
        rows.sort(key=lambda r: r["predicted_relative_energy_kcal_per_mol"])
        return rows
