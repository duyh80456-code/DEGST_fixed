"""Unstructured masks: allocation, prune/regrow, gradual pruning, schedules.

A single ``MaskManager`` serves every unstructured method (DEGST and its
ablations, RigL, GMP, SNIP, movement, Taylor/Fisher), so the methods differ only
in the score they pass in and in the schedule the trainer drives -- which is what
the C0 ablation needs.

Masked weights stay in the forward pass as exact zeros, so the dense gradient is
available at inactive positions for regrowth. Gradients of inactive weights are
zeroed before the optimiser step, so their momentum stays zero; newly grown
weights start at zero (RigL convention).
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn


def prunable_layers(model: nn.Module) -> Dict[str, nn.Module]:
    """All conv/linear layers; their ``.weight`` tensors are what sparsity counts."""
    return {n: m for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))}


def erk_densities(layers: Dict[str, nn.Module], density: float) -> Dict[str, float]:
    """Erdos-Renyi-Kernel allocation (Evci et al., 2020), dense layers capped at 1."""
    shapes = {n: m.weight.shape for n, m in layers.items()}
    numel = {n: s.numel() for n, s in shapes.items()}
    total = sum(numel.values())
    dense = set()
    while True:
        budget = density * total - sum(numel[n] for n in dense)
        raw = {n: sum(s) / s.numel() for n, s in shapes.items() if n not in dense}
        eps = budget / sum(raw[n] * numel[n] for n in raw)
        over = [n for n in raw if eps * raw[n] > 1]
        if not over:
            break
        dense.update(over)
    return {n: 1.0 if n in dense else eps * raw[n] for n in shapes}


def cubic_sparsity(t, t0, t1, s_final, s_init=0.0):
    """Zhu & Gupta (2017) gradual schedule."""
    if t <= t0:
        return s_init
    if t >= t1:
        return s_final
    return s_final + (s_init - s_final) * (1 - (t - t0) / (t1 - t0)) ** 3


def cosine_fraction(t, t0, t1, alpha):
    """Fraction of active weights swapped at step t (RigL cosine decay)."""
    if t >= t1:
        return 0.0
    return alpha / 2 * (1 + math.cos(math.pi * max(0, t - t0) / max(1, t1 - t0)))


def _bottom_k(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k smallest scores; ties broken by index (stable sort)."""
    return torch.sort(scores, stable=True).indices[:k]


def _top_k(scores: torch.Tensor, k: int) -> torch.Tensor:
    return torch.sort(scores, descending=True, stable=True).indices[:k]


class MaskManager:
    def __init__(self, model: nn.Module, allocation='global', min_density=0.01):
        self.model = model
        self.allocation = allocation
        self.min_density = min_density
        self.layers = prunable_layers(model)
        self.masks: Dict[str, torch.Tensor] = {
            n: torch.ones_like(m.weight, dtype=torch.bool) for n, m in self.layers.items()}
        self.dense_numel = sum(m.weight.numel() for m in self.layers.values())

    # ------------------------------------------------------------------ basics
    def weight(self, n):
        return self.layers[n].weight

    @torch.no_grad()
    def apply(self):
        for n, m in self.layers.items():
            m.weight.mul_(self.masks[n])

    @torch.no_grad()
    def mask_grads(self):
        for n, m in self.layers.items():
            if m.weight.grad is not None:
                m.weight.grad.mul_(self.masks[n])

    def nnz(self) -> int:
        return int(sum(m.sum().item() for m in self.masks.values()))

    def sparsity(self, reference_numel: Optional[int] = None) -> float:
        ref = reference_numel or self.dense_numel
        return 1.0 - self.nnz() / ref

    @torch.no_grad()
    def reset_momentum(self, optimizer, changed: Dict[str, torch.Tensor]):
        """Zero momentum for positions whose mask just changed (pruned or grown)."""
        for n, idx in changed.items():
            p = self.weight(n)
            st = optimizer.state.get(p, {})
            buf = st.get('momentum_buffer')
            if buf is not None:
                buf.view(-1)[idx] = 0

    # ------------------------------------------------------------------ allocation
    def layer_densities(self, density: float) -> Dict[str, float]:
        if self.allocation == 'erk':
            return erk_densities(self.layers, density)
        return {n: density for n in self.layers}

    @torch.no_grad()
    def init_random(self, sparsity: float, generator=None):
        """Random mask with per-layer density from the allocation (global -> uniform)."""
        dens = self.layer_densities(1 - sparsity)
        for n, m in self.layers.items():
            k = max(1, int(round(dens[n] * m.weight.numel())))
            perm = torch.randperm(m.weight.numel(), generator=generator)[:k].to(m.weight.device)
            mask = torch.zeros(m.weight.numel(), dtype=torch.bool, device=m.weight.device)
            mask[perm] = True
            self.masks[n] = mask.view_as(m.weight)
        self.apply()

    @torch.no_grad()
    def init_from_scores(self, scores: Dict[str, torch.Tensor], sparsity: float):
        """Keep the highest-scoring weights so that overall sparsity equals ``sparsity``."""
        self.masks = {n: torch.ones_like(self.masks[n]) for n in self.masks}
        target_nnz = int(round((1 - sparsity) * self.dense_numel))
        self._prune_to(scores, target_nnz)
        self.apply()

    # ------------------------------------------------------------------ pruning
    def _floor(self, n):
        return max(1, int(math.ceil(self.min_density * self.masks[n].numel())))

    @torch.no_grad()
    def _prune_to(self, scores, target_nnz) -> Dict[str, torch.Tensor]:
        """Remove active weights with the lowest scores until nnz == target_nnz."""
        k_total = self.nnz() - target_nnz
        if k_total <= 0:
            return {}
        if self.allocation == 'global':
            return self._prune_global(scores, k_total)
        dens = self.layer_densities(target_nnz / self.dense_numel)
        removed = {}
        for n in self.layers:
            tgt = max(self._floor(n), int(round(dens[n] * self.masks[n].numel())))
            k = int(self.masks[n].sum().item()) - tgt
            if k > 0:
                removed[n] = self._prune_layer(n, scores[n], k)
        return removed

    def _prune_layer(self, n, score, k):
        flat_m = self.masks[n].view(-1)
        act = flat_m.nonzero().squeeze(1)
        sel = act[_bottom_k(score.reshape(-1)[act].float(), k)]
        flat_m[sel] = False
        return sel

    def _prune_global(self, scores, k_total):
        names, cand, owners, pos = list(self.layers), [], [], []
        for i, n in enumerate(names):
            flat_m = self.masks[n].view(-1)
            act = flat_m.nonzero().squeeze(1)
            s = scores[n].reshape(-1)[act].float().clone()
            # per-layer floor: the top-`floor` active weights of each layer are never pruned
            f = min(self._floor(n), act.numel())
            if f > 0:
                s[_top_k(s, f)] = float('inf')
            cand.append(s)
            pos.append(act)
            owners.append(torch.full_like(act, i))
        cand, pos, owners = torch.cat(cand), torch.cat(pos), torch.cat(owners)
        k_total = min(k_total, int(torch.isfinite(cand).sum().item()))
        sel = _bottom_k(cand, k_total)
        removed = {}
        for i, n in enumerate(names):
            idx = pos[sel[owners[sel] == i]]
            if idx.numel():
                self.masks[n].view(-1)[idx] = False
                removed[n] = idx
        return removed

    @torch.no_grad()
    def prune_to_sparsity(self, scores, sparsity) -> Dict[str, torch.Tensor]:
        """Gradual pruning step (prune only)."""
        removed = self._prune_to(scores, int(round((1 - sparsity) * self.dense_numel)))
        self.apply()
        return removed

    @torch.no_grad()
    def prune_regrow(self, scores, grow_scores, fraction) -> Dict[str, int]:
        """Swap ``fraction`` of active weights: prune lowest ``scores``, grow highest ``grow_scores``.

        Sparsity is unchanged. With global allocation both the prune and the grow
        sets are chosen globally; otherwise per layer.
        """
        n_act = self.nnz()
        k = int(fraction * n_act)
        if k <= 0:
            return {'pruned': 0, 'grown': 0, 'changed': {}}
        before = {n: m.clone() for n, m in self.masks.items()}
        if self.allocation == 'global':
            removed = self._prune_global(scores, k)
            grown = self._grow_global(grow_scores, sum(v.numel() for v in removed.values()), before)
        else:
            removed, grown = {}, {}
            for n in self.layers:
                n_act = int(before[n].sum().item())
                # a layer can swap at most as many weights as it has free slots (dense ERK layers: none)
                kl = min(int(fraction * n_act), n_act - self._floor(n), before[n].numel() - n_act)
                if kl > 0:
                    removed[n] = self._prune_layer(n, scores[n], kl)
                    grown[n] = self._grow_layer(n, grow_scores[n], kl, before[n])
        changed = {}
        for n in self.layers:
            parts = [x for x in (removed.get(n), grown.get(n)) if x is not None and x.numel()]
            if parts:
                changed[n] = torch.cat(parts)
        # newly grown weights start at zero
        for n, idx in grown.items():
            self.weight(n).view(-1)[idx] = 0
        self.apply()
        return {'pruned': sum(v.numel() for v in removed.values()),
                'grown': sum(v.numel() for v in grown.values()), 'changed': changed}

    def _grow_layer(self, n, gscore, k, before):
        flat_m = self.masks[n].view(-1)
        # candidates: inactive before this update (just-pruned weights are not regrown at once)
        inact = (~before.view(-1)).nonzero().squeeze(1)
        k = min(k, inact.numel())
        sel = inact[_top_k(gscore.reshape(-1)[inact].float(), k)]
        flat_m[sel] = True
        return sel

    def _grow_global(self, gscores, k, before):
        names, cand, owners, pos = list(self.layers), [], [], []
        for i, n in enumerate(names):
            inact = (~before[n].view(-1)).nonzero().squeeze(1)
            cand.append(gscores[n].reshape(-1)[inact].float())
            pos.append(inact)
            owners.append(torch.full_like(inact, i))
        cand, pos, owners = torch.cat(cand), torch.cat(pos), torch.cat(owners)
        sel = _top_k(cand, min(k, cand.numel()))
        grown = {}
        for i, n in enumerate(names):
            idx = pos[sel[owners[sel] == i]]
            if idx.numel():
                self.masks[n].view(-1)[idx] = True
                grown[n] = idx
        return grown
