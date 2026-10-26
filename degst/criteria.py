"""Weight-level importance criteria, all evaluated inside the same prune/regrow loop.

    magnitude          |w|                               (RigL / GMP drop criterion)
    entropy            H(A_in)                           (entropy alone, C0 config B)
    entropy_magnitude  H(A_in) * |w|                     (DEGST, C0 config C)
    taylor             mean_t |w * g|                    (first-order Taylor, Molchanov et al.)
    fisher             mean_t (w * g)^2 / 2              (empirical-Fisher OBD saliency)
    movement           -sum_t w * g                      (Sanh et al., cumulative)
    gradient           mean_t |g|
    random             uniform noise                     (control: no criterion; own RNG, seeded per run)
    energy_magnitude   |w| * sqrt(E[a_in^2])             (S1: Wanda's score, Sun et al. 2024, per input channel)
    composite          |w| * (1 + beta * u)              (S3: magnitude-anchored; u = a data signal min-max
                                                          normalised to [0, 1] per layer: sqrt(E[a_in^2]) per
                                                          input channel ('energy') or mean_t g^2 ('fisher');
                                                          beta = 0 is |w|, beta -> inf ranks by u within a layer)

Note on Fisher: 'fisher' above is already the OBD form F_hat * w^2 / 2 (with F_hat = mean_t g^2 from
mini-batch gradients), not F_hat alone.
    snip              |w * g| at initialisation         (one-shot)
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
COMPOSITE_SIGNALS = ('energy', 'fisher')


def _minmax(u: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Rescale to [0, 1] over the whole tensor (one layer); a constant tensor maps to 0."""
    lo, hi = u.min(), u.max()
    return (u - lo) / (hi - lo).clamp_min(eps)


class ScoreState:
    def __init__(self, layers: Dict[str, torch.nn.Module], criterion: str, seed: int = 0,
                 beta: float = 0.5, signal: str = 'energy'):
        if criterion == 'composite' and signal not in COMPOSITE_SIGNALS:
            raise ValueError(f'composite signal must be one of {COMPOSITE_SIGNALS}, not {signal!r}')
        self.layers, self.criterion = layers, criterion
        self.beta, self.signal = float(beta), signal
        # gradient statistics: the four accumulated criteria, and the composite with the Fisher signal (g^2)
        self.accumulates = criterion in NEEDS_ACCUM or (criterion == 'composite' and signal == 'fisher')
        self.acc = {n: torch.zeros_like(m.weight) for n, m in layers.items()} if self.accumulates else {}
        self.count = 0
        self.seed, self._gen = seed, None   # 'random' only: a private generator leaves the global RNG alone

    @property
    def activation_stat(self):
        """Per-input-channel statistic the score needs from the forward hooks: 'entropy', 'energy' or None."""
        if self.criterion in NEEDS_ENTROPY:
            return 'entropy'
        if self.criterion == 'energy_magnitude' or (self.criterion == 'composite' and self.signal == 'energy'):
            return 'energy'
        return None

    def _random(self, w: torch.Tensor) -> torch.Tensor:
        if self._gen is None:
            self._gen = torch.Generator(device=w.device).manual_seed(self.seed)
        return torch.rand(w.shape, generator=self._gen, device=w.device)

    @torch.no_grad()
    def accumulate(self):
        if not self.accumulates:
            return
        for n, m in self.layers.items():
            g = m.weight.grad
            if g is None:
                continue
            w = m.weight
            if self.criterion == 'composite':        # Fisher signal: F_hat = mean_t g^2 (w enters via |w|)
                self.acc[n] += g * g
            elif self.criterion == 'taylor':
                self.acc[n] += (w * g).abs()
            elif self.criterion == 'fisher':
                self.acc[n] += 0.5 * (w * g) ** 2
            elif self.criterion == 'movement':
                self.acc[n] -= w * g
            elif self.criterion == 'gradient':
                self.acc[n] += g.abs()
        self.count += 1

    @torch.no_grad()
    def scores(self, stats: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Drop scores per layer. ``stats`` is the per-input-channel statistic named by
        ``activation_stat`` (entropy or mean squared activation), as recorded by EntropyTracker."""
        c, out = self.criterion, {}
        for n, m in self.layers.items():
            w = m.weight.detach()
            if self.activation_stat is not None:
                if stats is None or n not in stats:
                    raise RuntimeError(f'no {self.activation_stat} recorded for layer {n}')
                per_weight = expand_to_weight(stats[n].to(w.device), w, getattr(m, 'groups', 1))
            if c == 'magnitude':
                s = w.abs()
            elif c == 'random':
                s = self._random(w)
            elif c in NEEDS_ENTROPY:
                s = per_weight.clone() if c == 'entropy' else per_weight * w.abs()
            elif c == 'energy_magnitude':
                s = w.abs() * per_weight.clamp_min(0).sqrt()
            elif c == 'composite':
                if self.signal == 'energy':
                    u = _minmax(per_weight.clamp_min(0).sqrt())
                else:
                    u = _minmax(self.acc[n] / max(1, self.count))
                s = w.abs() * (1.0 + self.beta * u)
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
        if self.accumulates and self.criterion != 'movement':
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
