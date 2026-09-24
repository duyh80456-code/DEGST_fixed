"""Weight-level importance criteria, all evaluated inside the same prune/regrow loop.

    magnitude          |w|                               (RigL / GMP drop criterion)
    entropy            H(A_in)                           (entropy alone, C0 config B)
    entropy_magnitude  H(A_in) * |w|                     (DEGST, C0 config C)
    taylor             mean_t |w * g|                    (first-order Taylor, Molchanov et al.)
    fisher             mean_t (w * g)^2 / 2              (empirical-Fisher OBD saliency)
    movement           -sum_t w * g                      (Sanh et al., cumulative)
    gradient           mean_t |g|
    snip               |w * g| at initialisation         (one-shot)
    eltp               |w * g| * (1 - E_i / log n)       (pre-revision "DEGST" score, instantaneous g;
                                                          E_i = -p_i log p_i, p_i = |w_i| / sum_layer |w|)

H(A_in) is the entropy of the *input* channel a weight reads from; see
``entropy.expand_to_weight``. Accumulated criteria are updated every step from
the dense gradient (before inactive gradients are masked) and averaged over the
steps since the last topology update.
"""
import math
from typing import Dict

import torch

from .entropy import expand_to_weight

NEEDS_ENTROPY = {'entropy', 'entropy_magnitude'}
NEEDS_ACCUM = {'taylor', 'fisher', 'movement', 'gradient'}


class ScoreState:
    def __init__(self, layers: Dict[str, torch.nn.Module], criterion: str):
        self.layers, self.criterion = layers, criterion
        self.acc = {n: torch.zeros_like(m.weight) for n, m in layers.items()} if criterion in NEEDS_ACCUM else {}
        self.count = 0

    @torch.no_grad()
    def accumulate(self):
        if self.criterion not in NEEDS_ACCUM:
            return
        for n, m in self.layers.items():
            g = m.weight.grad
            if g is None:
                continue
            w = m.weight
            if self.criterion == 'taylor':
                self.acc[n] += (w * g).abs()
            elif self.criterion == 'fisher':
                self.acc[n] += 0.5 * (w * g) ** 2
            elif self.criterion == 'movement':
                self.acc[n] -= w * g
            elif self.criterion == 'gradient':
                self.acc[n] += g.abs()
        self.count += 1

    @torch.no_grad()
    def scores(self, entropy: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        c, out = self.criterion, {}
        for n, m in self.layers.items():
            w = m.weight.detach()
            if c == 'magnitude':
                s = w.abs()
            elif c in NEEDS_ENTROPY:
                if entropy is None or n not in entropy:
                    raise RuntimeError(f'no entropy recorded for layer {n}')
                h = expand_to_weight(entropy[n].to(w.device), w, getattr(m, 'groups', 1))
                s = h.clone() if c == 'entropy' else h * w.abs()
            elif c == 'snip':
                s = (w * m.weight.grad).abs()
            elif c == 'eltp':
                s = eltp_score(w, m.weight.grad)
            elif c == 'movement':
                s = self.acc[n].clone()
            else:
                s = self.acc[n] / max(1, self.count)
            out[n] = s
        return out

    def reset(self):
        """Start a new averaging window (movement keeps its cumulative sum)."""
        if self.criterion in ('taylor', 'fisher', 'gradient'):
            for a in self.acc.values():
                a.zero_()
            self.count = 0


@torch.no_grad()
def eltp_score(w: torch.Tensor, g: torch.Tensor, eps=1e-12) -> torch.Tensor:
    """Score used by the pre-revision code (archive/legacy_code.zip: utils/dynamic_trainer.py, utils/pruning.py).

    The "entropy" here is that of the layer's normalised weight magnitudes, not of
    activations; it is kept only to reproduce and compare with the earlier results.
    """
    if g is None:
        return torch.zeros_like(w)
    flat = w.abs().flatten()
    s = flat.sum()
    p = flat / (s + eps) if s > 0 else torch.full_like(flat, 1.0 / flat.numel())
    e = (-p * torch.log(p + eps)).view_as(w)
    h_max = math.log(w.numel()) if w.numel() > 1 else 1.0
    return (w * g).abs() * (1.0 - e / (h_max + eps))


def grad_scores(layers) -> Dict[str, torch.Tensor]:
    """Regrowth criterion: dense gradient magnitude (RigL)."""
    return {n: m.weight.grad.detach().abs() if m.weight.grad is not None else torch.zeros_like(m.weight)
            for n, m in layers.items()}
