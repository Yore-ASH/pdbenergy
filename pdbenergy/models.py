"""Neural network architectures for predicting conformational energy.

Two models are provided on purpose:

``SchNetRegressor``
    A message-passing graph neural network (SchNet, Schuett et al. 2017) that
    consumes the 3D structure directly.  Energy is assembled as a **sum of
    per-atom contributions**, which makes the prediction *extensive*: doubling
    the size of the system roughly doubles the energy, as physics requires.

``MLPRegressor``
    A plain multi-layer perceptron over the hand-crafted global descriptors from
    :mod:`pdbenergy.features`.  This is the honest baseline: it shows how much
    accuracy actually comes from the graph architecture rather than from the
    training setup.

Both models are invariant to atom permutation (aggregation is a sum) and to
translation/rotation (they only ever see distances / invariant scalars).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .features import DESCRIPTOR_DIM, MAX_Z

#: Elements whose per-molecule counts become global features.  Composition is
#: the single biggest determinant of an *absolute* energy, so handing the model
#: the raw atom counts saves it from having to count atoms through message
#: passing.  Index 5 collects "everything else".
COMPOSITION_ELEMENTS: tuple[int, ...] = (1, 6, 7, 8, 16)
N_COMPOSITION_FEATURES = len(COMPOSITION_ELEMENTS) + 1


def zero_initialise_last_linear(module: nn.Sequential) -> None:
    """Zero the final Linear of a head so the untrained model predicts 0.

    See :meth:`SchNetRegressor._zero_initialise_readout` for why this matters.
    It is applied to *both* architectures: a baseline that starts from a wildly
    miscalibrated output would not be a fair comparison, it would just be a
    badly configured baseline.
    """
    last = module[-1]
    if isinstance(last, nn.Linear):
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Segment sum - the aggregation step of every message-passing layer.

    This is a *permutation-invariant* operation: shuffling the atoms shuffles the
    summands but not the result.  ``index_add_`` accumulates ``src[e]`` into row
    ``index[e]``, which is exactly a scatter-add, and it needs no external graph
    library.
    """
    out = torch.zeros((dim_size, *src.shape[1:]), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def composition_features(z: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    """Per-molecule atom counts for the main elements, normalised by molecule size."""
    one_hot = torch.zeros((z.shape[0], N_COMPOSITION_FEATURES), dtype=torch.float32, device=z.device)
    for column, atomic_number in enumerate(COMPOSITION_ELEMENTS):
        one_hot[:, column] = (z == atomic_number).float()
    one_hot[:, -1] = (~torch.isin(z, torch.tensor(COMPOSITION_ELEMENTS, device=z.device))).float()
    counts = scatter_sum(one_hot, batch, n_graphs)
    return counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)


class GaussianRBFExpansion(nn.Module):
    """Expand a distance into a Gaussian radial basis, with a cutoff envelope.

    $$e_k(r) = \\exp\\left(-\\gamma\\,(r - \\mu_k)^2\\right),\\qquad
      \\gamma = \\frac{1}{2\\Delta^2},\\ \\Delta = \\frac{r_{max}-r_{min}}{K-1}$$

    and then multiply by the cosine window
    ``0.5 * (cos(pi r / r_c) + 1)`` so the message smoothly vanishes at the
    cutoff.  A hard cutoff would make the predicted energy *discontinuous* in
    the coordinates whenever a pair crosses ``r_c``, which is unacceptable for
    a potential energy surface.

    Doing this on the raw distance tensor (one float per edge) rather than
    storing dense RBF vectors is both a memory and a speed win, and it keeps the
    basis parameters next to the weights that use them.
    """

    def __init__(self, n_rbf: int, cutoff: float, rbf_start: float, rbf_end: float):
        super().__init__()
        self.n_rbf = int(n_rbf)
        self.cutoff = float(cutoff)

        # The basis must cover the cutoff.  If it does not, every edge longer than
        # rbf_end has *all* basis functions underflow to zero (the Gaussians are
        # narrow: at n_rbf=64 over 0-3 A the spacing is 0.048 A, so a distance 1 A
        # past the last centre gives exp(-220)).  Those edges then reach the
        # network carrying only the (zero) bond flag - silent information loss
        # that looks like "a coarser model" rather than a broken config.
        if rbf_end < cutoff * 0.98:
            raise ValueError(
                f"features.rbf_end ({rbf_end:g}) is below features.cutoff ({cutoff:g}). "
                f"Edges between {rbf_end:g} and {cutoff:g} A would carry all-zero radial "
                f"features. Set rbf_end >= cutoff (e.g. rbf_end={cutoff:g}), or widen the "
                f"basis; to get finer resolution, raise n_rbf instead of shrinking the range."
            )

        centers = torch.linspace(rbf_start, rbf_end, self.n_rbf, dtype=torch.float32)
        width = (rbf_end - rbf_start) / max(1, self.n_rbf - 1)
        self.register_buffer("centers", centers, persistent=False)
        self.gamma = 1.0 / (2.0 * width * width)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        r = distances.reshape(-1, 1)
        basis = torch.exp(-self.gamma * (r - self.centers.reshape(1, -1)) ** 2)
        # Cosine cutoff envelope: 0.5*(cos(pi*r/rc)+1) for r < rc, else 0.
        clamped = torch.clamp(r, max=self.cutoff)
        envelope = 0.5 * (torch.cos(torch.pi * clamped / self.cutoff) + 1.0)
        envelope = torch.where(r < self.cutoff, envelope, torch.zeros_like(envelope))
        return basis * envelope


class RBFFilter(nn.Module):
    """Learns a per-distance multiplicative filter for messages.

    ``W(r_ij)`` maps the radial-basis description of an edge to a
    ``hidden_dim``-wide vector that modulates the sender's features.  This is
    what lets one shared weight matrix behave differently at 1.5 A (a covalent
    bond) and at 4.0 A (a non-bonded contact).
    """

    def __init__(self, n_rbf: int, hidden_dim: int, extra_edge_dim: int = 0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_rbf + extra_edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.net(edge_attr)


class InteractionBlock(nn.Module):
    """One SchNet interaction: continuous-filter convolution + residual update."""

    def __init__(self, hidden_dim: int, n_rbf: int, extra_edge_dim: int = 0):
        super().__init__()
        self.filter = RBFFilter(n_rbf, hidden_dim, extra_edge_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        n_atoms: int,
    ) -> torch.Tensor:
        receiver, sender = edge_index[0], edge_index[1]
        weights = self.filter(edge_attr)              # (E, H)
        messages = x[sender] * weights                # (E, H)
        aggregated = scatter_sum(messages, receiver, n_atoms)
        return x + self.activation(self.linear(aggregated))


class SchNetRegressor(nn.Module):
    """Message-passing network mapping a 3D structure to a scalar energy.

    Architecture (each step is explained in ``TeachFlow.md``):

    1. **embedding** - atomic number -> ``hidden_dim`` vector;
    2. **n_interactions x InteractionBlock** - each atom accumulates information
       from neighbours within the cutoff, so after k blocks it "sees" roughly
       ``k * cutoff`` of its environment;
    3. **readout** - per-atom MLP producing that atom's energy contribution;
    4. **sum pooling** - total energy = sum of atom contributions (extensive);
    5. **global residual head** - a small MLP over the pooled energy plus the
       molecule's composition, which a pure sum cannot represent on its own.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        n_rbf: int,
        *,
        extra_edge_dim: int = 1,
        rbf_start: float = 0.0,
        rbf_end: float = 5.0,
        cutoff: float = 4.5,
    ):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden_dim
        self.extra_edge_dim = int(extra_edge_dim)
        self.rbf = GaussianRBFExpansion(n_rbf, cutoff, rbf_start, rbf_end)
        self.embedding = nn.Embedding(MAX_Z, hidden)
        self.interactions = nn.ModuleList(
            InteractionBlock(hidden, n_rbf, extra_edge_dim)
            for _ in range(cfg.n_interactions)
        )
        self.atom_readout = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.global_head = nn.Sequential(
            nn.Linear(1 + N_COMPOSITION_FEATURES, hidden // 2),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden // 2, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self._zero_initialise_readout()

    def _zero_initialise_readout(self) -> None:
        """Start the network predicting exactly zero (the target mean).

        Why this matters
        ----------------
        The energy is a **sum over ~700 atoms**.  If each atom's readout starts
        with random weights, the sum of 700 small random numbers is not small:
        the untrained model predicts hundreds of kcal/mol away from the mean.
        Measured on this dataset, that cost the first epochs entire *hundreds* of
        kcal/mol of validation MAE while the optimiser dragged the output scale
        back down.

        Zeroing the final layer of the atom readout and of the global head makes
        the initial prediction exactly the standardised mean (0), i.e. the
        constant predictor - the correct starting point for a regression whose
        target has been standardised.  Learning then only has to add structure,
        never to undo random scale.
        """
        for head in (self.atom_readout, self.global_head):
            zero_initialise_last_linear(head)

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        n_graphs: int,
        **_: object,
    ) -> torch.Tensor:
        n_atoms = z.shape[0]
        # Edge layout: column 0 is the distance, the rest are extra channels.
        # Expand the distance into the radial basis once per forward pass.
        distances = edge_attr[:, 0]
        expanded = self.rbf(distances)
        if self.extra_edge_dim > 0:
            expanded = torch.cat([expanded, edge_attr[:, 1:1 + self.extra_edge_dim]], dim=-1)

        x = self.embedding(z.clamp(0, MAX_Z - 1))
        for block in self.interactions:
            x = block(x, edge_index, expanded, n_atoms)

        per_atom = self.atom_readout(x).squeeze(-1)             # (N,)
        pooled = scatter_sum(per_atom, batch, n_graphs)         # (B,)
        composition = composition_features(z, batch, n_graphs)  # (B, F)
        correction = self.global_head(torch.cat([pooled.unsqueeze(-1), composition], dim=-1))
        return pooled + correction.squeeze(-1)


class MLPRegressor(nn.Module):
    """Baseline: fully-connected network over hand-crafted descriptors."""

    def __init__(self, cfg: ModelConfig, *, in_dim: int = DESCRIPTOR_DIM):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(max(1, cfg.n_layers_mlp)):
            layers += [nn.Linear(dim, cfg.hidden_dim), nn.SiLU(), nn.Dropout(cfg.dropout)]
            dim = cfg.hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)
        # Same treatment as the graph network: start at the target mean.
        zero_initialise_last_linear(self.net)

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def edge_dims_for(feature_cfg) -> dict:
    """Model-construction kwargs implied by a :class:`FeatureConfig`.

    The model must be built with exactly the edge-feature layout *and* the same
    radial basis the featuriser assumed, so both are derived from one place.
    """
    return {
        "n_rbf": int(getattr(feature_cfg, "n_rbf", 48)),
        "extra_edge_dim": 1 if getattr(feature_cfg, "use_bond_feature", True) else 0,
        "rbf_start": float(getattr(feature_cfg, "rbf_start", 0.0)),
        "rbf_end": float(getattr(feature_cfg, "rbf_end", 5.0)),
        "cutoff": float(getattr(feature_cfg, "cutoff", 4.5)),
    }


def build_model(cfg: ModelConfig, **edge_kwargs) -> nn.Module:
    """Factory honouring ``cfg.kind``; ``edge_kwargs`` come from :func:`edge_dims_for`."""
    if cfg.kind == "schnet":
        return SchNetRegressor(cfg, **edge_kwargs)
    if cfg.kind == "mlp":
        return MLPRegressor(cfg)
    raise ValueError(f"unknown model kind: {cfg.kind!r}")


def count_parameters(model: nn.Module) -> int:
    """Number of trainable scalar parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
