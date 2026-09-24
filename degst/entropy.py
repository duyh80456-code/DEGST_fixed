"""Histogram estimate of per-channel activation entropy, collected with hooks.

H(A_c) = -sum_k p_k log p_k, with p_k the fraction of the channel's values that
fall in bin k of B equal-width bins spanning that channel's [min, max] in the
current mini-batch. A constant channel has H = 0; the upper bound is log B.
"""
import time
from typing import Dict, List

import torch
import torch.nn as nn


@torch.no_grad()
def channel_entropy(x: torch.Tensor, bins: int = 30, eps: float = 1e-8, max_samples: int = 0) -> torch.Tensor:
    """Entropy of each channel (dim 1) of ``x``; returns a float tensor of shape [C].

    ``max_samples > 0`` randomly subsamples that many values per channel, which
    bounds the cost on large early-layer feature maps.
    """
    c = x.shape[1]
    xc = x.detach().float().transpose(0, 1).reshape(c, -1)
    if max_samples and xc.shape[1] > max_samples:
        idx = torch.randint(0, xc.shape[1], (max_samples,), device=xc.device)
        xc = xc[:, idx]
    lo = xc.min(1, keepdim=True).values
    hi = xc.max(1, keepdim=True).values
    b = ((xc - lo) / (hi - lo + eps) * bins).long().clamp_(0, bins - 1)
    b += torch.arange(c, device=x.device).unsqueeze(1) * bins
    counts = torch.bincount(b.flatten(), minlength=c * bins).view(c, bins).float()
    p = counts / counts.sum(1, keepdim=True)
    return -(p * torch.log(p + eps)).sum(1)


class EntropyTracker:
    """Records per-channel entropy of module inputs (``mode='input'``) or outputs.

    Collection is off by default; the trainer switches it on for the last
    ``n_batches`` training steps before each topology update, so no extra forward
    passes are needed. ``reduce()`` aggregates the per-batch values by mean or
    median and clears the buffer. Time spent in the hooks is accumulated in
    ``seconds`` (CUDA-synchronised) so the overhead can be reported (C2.5).
    """

    def __init__(self, modules: Dict[str, nn.Module], mode='input', bins=30, reduction='mean',
                 max_samples=0, gate_fn=None):
        self.bins, self.reduction, self.max_samples = bins, reduction, max_samples
        self.active = False
        self.buf: Dict[str, List[torch.Tensor]] = {k: [] for k in modules}
        self.seconds = 0.0
        self.handles = []
        for name, m in modules.items():
            if mode == 'input':
                h = m.register_forward_pre_hook(self._make_pre(name))
            else:
                h = m.register_forward_hook(self._make_post(name))
            self.handles.append(h)

    def _record(self, name, x):
        if not self.active or not torch.is_grad_enabled():
            return
        sync = x.is_cuda
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.buf[name].append(channel_entropy(x, self.bins, max_samples=self.max_samples))
        if sync:
            torch.cuda.synchronize()
        self.seconds += time.perf_counter() - t0

    def _make_pre(self, name):
        def hook(_m, inp):
            self._record(name, inp[0])
        return hook

    def _make_post(self, name):
        def hook(_m, _inp, out):
            self._record(name, out)
        return hook

    def reduce(self) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in self.buf.items():
            if v:
                s = torch.stack(v)
                out[k] = s.median(0).values if self.reduction == 'median' else s.mean(0)
            self.buf[k] = []
        return out

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def expand_to_weight(h_in: torch.Tensor, weight: torch.Tensor, groups: int = 1) -> torch.Tensor:
    """Broadcast input-channel entropy onto a conv/linear weight tensor.

    For a grouped convolution, weight[o, i] reads input channel
    (o // out_per_group) * in_per_group + i.
    """
    c_out, c_in_g = weight.shape[0], weight.shape[1]
    if groups == 1:
        h = h_in.view(1, -1).expand(c_out, -1)
    else:
        out_per_group = c_out // groups
        grp = torch.arange(c_out, device=h_in.device) // out_per_group
        idx = grp.unsqueeze(1) * c_in_g + torch.arange(c_in_g, device=h_in.device).unsqueeze(0)
        h = h_in[idx]
    return h.view(c_out, c_in_g, *([1] * (weight.dim() - 2))).expand_as(weight)
