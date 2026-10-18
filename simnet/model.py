"""Fixed-sparsity binary layers and initialisation schemes."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Exact k-winners-take-all mask (0/1, exactly k ones per row)."""
    idx = torch.topk(scores, k, dim=1).indices
    return torch.zeros_like(scores).scatter_(1, idx, 1.0)


def soft_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Euclidean projection onto {0 <= p <= 1, sum(p)=k}, row-wise.

    Used only for diagnostics / optional soft training.  At inference the
    model uses the hard top-k mask, so the number of active neurons is exact.
    """
    lo = scores.min(dim=1, keepdim=True).values - 1.0
    hi = scores.max(dim=1, keepdim=True).values + 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        p = torch.clamp(scores - mid, 0.0, 1.0)
        too_much = p.sum(1, keepdim=True) > k
        hi = torch.where(too_much, mid, hi)
        lo = torch.where(too_much, lo, mid)
    return torch.clamp(scores - (lo + hi) / 2, 0.0, 1.0)


class SparseStack(nn.Module):
    """A stack of linear layers followed by exact top-k sparsification.

    Layer 0 is the fixed input encoder (not a module).  ``widths[0]`` must be
    240, the input-code width.  For layer i >= 1, ``k_i = round(width_i *
    sparsity_i)`` neurons are active.
    """

    def __init__(
        self,
        widths: tuple[int, ...] = (240, 300, 400, 600, 1200),
        sparsities: tuple[float, ...] = (0.5, 0.4, 0.3, 0.2, 0.1),
        seed: int = 0,
        init: str = "gaussian",
        center: bool = True,
    ):
        super().__init__()
        if len(widths) != len(sparsities):
            raise ValueError("widths and sparsities must have the same length")
        self.widths = tuple(widths)
        self.sparsities = tuple(sparsities)
        self.ks = [max(1, int(round(w * s))) for w, s in zip(widths, sparsities)]
        self.init = init

        g = torch.Generator().manual_seed(seed)
        layers: list[nn.Linear] = []
        for i in range(1, len(widths)):
            lin = nn.Linear(widths[i - 1], widths[i], bias=True)
            if init == "gaussian":
                std = 1.0 / math.sqrt(widths[i - 1])
                with torch.no_grad():
                    lin.weight.normal_(0.0, std, generator=g)
                    if center:
                        # All input codes have a fixed number of active bits.
                        # Subtracting the row mean removes a constant per-unit
                        # offset that would otherwise dominate the top-k.
                        lin.weight.sub_(lin.weight.mean(dim=1, keepdim=True))
                    lin.bias.zero_()
            elif init != "zero":
                raise ValueError(f"unknown init: {init}")
            layers.append(lin)
        self.layers = nn.ModuleList(layers)

    # ------------------------------------------------------------------
    def forward(
        self,
        code: torch.Tensor,
        hard: bool = True,
        return_scores: bool = False,
    ):
        """Return the list of layer patterns [input, L2, L3, L4, L5]."""
        hs: list[torch.Tensor] = [code]
        scores: list[torch.Tensor] = []
        h = code
        for lin, k in zip(self.layers, self.ks[1:]):
            s = lin(h)
            if hard:
                m = topk_mask(s, k)
                if self.training:
                    # Straight-through: forward is exactly binary; backward
                    # treats the mask as the identity.
                    h = m + s - s.detach()
                else:
                    h = m
            else:
                h = soft_topk(s, k)
            hs.append(h)
            scores.append(s)
        if return_scores:
            return hs, scores
        return hs

    def forward_to(self, code: torch.Tensor, upto: int) -> torch.Tensor:
        """Run through the first ``upto`` linear layers (0 = input code)."""
        h = code
        for i in range(upto):
            s = self.layers[i](h)
            h = topk_mask(s, self.ks[i + 1])
        return h

    # ------------------------------------------------------------------
    @torch.no_grad()
    def active_counts(self, code: torch.Tensor) -> list[torch.Tensor]:
        hs = self(code, hard=True)
        return [h.sum(dim=1).to(torch.long) for h in hs]


def _cpu_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


@torch.no_grad()
def permutation_init(
    model: SparseStack,
    seed: int = 0,
    noise: float = 0.2,
    extra_weight: float = 0.7,
) -> list[torch.Tensor]:
    """Coverage-preserving random sparse wiring.

    For each layer, the first ``in_features`` output neurons receive a random
    permutation of all input neurons (so every input feature is observable).
    Any extra output neurons (when ``out > in``) receive one random input
    neuron with a slightly smaller weight.  Weights are random and are then
    refined by backpropagation.  The returned masks keep the wiring sparse.
    """
    g = _cpu_generator(seed)
    masks: list[torch.Tensor] = []
    for lin in model.layers:
        out, inp = lin.weight.shape
        lin.weight.zero_()
        lin.bias.zero_()
        perm = torch.randperm(inp, generator=g)
        w = 1.0 + noise * torch.randn(inp, generator=g)
        w = w.clamp_min(0.05).to(lin.weight.device)
        lin.weight[torch.arange(inp, device=lin.weight.device), perm.to(lin.weight.device)] = w
        for j in range(inp, out):
            idx = int(torch.randint(0, inp, (1,), generator=g).item())
            val = max(0.05, extra_weight + noise * float(torch.randn(1, generator=g).item()))
            lin.weight[j, idx] = val
        masks.append((lin.weight != 0).clone())
    return masks


@torch.no_grad()
def random_one_to_one_init(
    model: SparseStack,
    seed: int = 0,
    weight_low: float = 0.5,
    weight_high: float = 1.5,
) -> list[torch.Tensor]:
    """Fully random one-input-per-neuron wiring (no coverage guarantee)."""
    g = _cpu_generator(seed)
    masks: list[torch.Tensor] = []
    for lin in model.layers:
        out, inp = lin.weight.shape
        lin.weight.zero_()
        lin.bias.zero_()
        for j in range(out):
            idx = int(torch.randint(0, inp, (1,), generator=g).item())
            val = float(torch.empty(1).uniform_(weight_low, weight_high, generator=g).item())
            lin.weight[j, idx] = val
        masks.append((lin.weight != 0).clone())
    return masks


@torch.no_grad()
def fan_in_init(
    model: SparseStack,
    fan: int = 2,
    seed: int = 0,
    noise: float = 0.1,
) -> list[torch.Tensor]:
    """Random fan-in ``fan`` wiring: each neuron reads ``fan`` previous neurons.

    fan=1 has the same fan-in as the main sparse relay (but without the
    coverage-preserving permutation); fan=2/4 interpolate between that and a
    fully-connected layer.  The number of input connections per neuron is
    fixed, so the mask keeps the layer sparse after training.
    """
    if fan < 1:
        raise ValueError("fan must be >= 1")
    g = _cpu_generator(seed)
    masks: list[torch.Tensor] = []
    for lin in model.layers:
        out, inp = lin.weight.shape
        if fan > inp:
            raise ValueError(f"fan={fan} exceeds layer input width {inp}")
        lin.weight.zero_()
        lin.bias.zero_()
        for j in range(out):
            idx = torch.randperm(inp, generator=g)[:fan]
            w = 1.0 / fan + noise * torch.randn(fan, generator=g)
            lin.weight[j, idx] = w.clamp_min(0.01).to(lin.weight.device)
        masks.append((lin.weight != 0).clone())
    return masks


@torch.no_grad()
def copy_init(model: SparseStack) -> None:
    """Exact isometric-copy oracle (only exact when width_i >= width_{i-1})."""
    for lin in model.layers:
        lin.weight.zero_()
        lin.bias.zero_()
        n = min(lin.in_features, lin.out_features)
        lin.weight[:n, :n] = torch.eye(n, device=lin.weight.device)


def nonzero_synapses(model: SparseStack) -> int:
    return int(sum((lin.weight != 0).sum().item() for lin in model.layers))


def total_neurons(model: SparseStack) -> int:
    return int(sum(model.widths))


def build_model(config: dict, seed: int = 0) -> SparseStack:
    return SparseStack(
        widths=tuple(config["widths"]),
        sparsities=tuple(config["sparsities"]),
        seed=seed,
    )
