"""Channel-level sparsity with *physical* removal (fixes C1.1).

Channels are handled per coupled group from Torch-Pruning's DepGraph (Fang et
al., 2023): in a ResNet the residual stream is one group spanning the stem,
every block's bn2 and the shortcut BNs, so its channels can only be removed
together. During training each group carries a gate vector that multiplies the
outputs of all BatchNorms in the group; a zero gate silences the channel
everywhere, which is exactly equivalent to removing it. At the end
``compact()`` folds the gates into the BNs and removes the silenced channels
with DepGraph, producing a smaller dense model whose params/FLOPs/latency are
measured directly.

Budgets are specified in *conv/linear weight parameters* so that a structured
model at "90% sparsity" has exactly the same weight count as an unstructured
one at 90% (up to one channel), which fixes C1.3/C1.4.
"""
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch_pruning as tp

from .entropy import EntropyTracker


class ChannelGroups:
    def __init__(self, model: nn.Module, example_input: torch.Tensor, classifier: nn.Module,
                 min_keep=0.05):
        self.model, self.min_keep = model, min_keep
        was_training = model.training
        model.eval()                      # tracing forward must not touch BN running stats
        self.DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_input)
        model.train(was_training)
        groups = list(self.DG.get_all_groups(ignored_layers=[classifier], root_module_types=[nn.Conv2d]))
        self.roots, self.size, self.bns, self.convs_out = [], [], [], []
        self.out_group, self.in_group = {}, {}
        for gi, g in enumerate(groups):
            root = g[0].dep
            self.roots.append((root.target.module, root.handler))
            self.size.append(len(g[0].idxs))
            bns, convs = [], []
            for dep, idxs in g:
                mod, h = dep.target.module, dep.handler
                is_out = self.DG.is_out_channel_pruning_fn(h)
                idx = torch.as_tensor(list(idxs), dtype=torch.long)
                if isinstance(mod, nn.BatchNorm2d) and is_out:
                    assert len(idx) == mod.num_features, 'partial BN coupling is not supported'
                    bns.append((mod, torch.argsort(idx)))
                if isinstance(mod, (nn.Conv2d, nn.Linear)):
                    if is_out:
                        self.out_group[mod] = gi
                        if isinstance(mod, nn.Conv2d):
                            convs.append((mod, idx))
                    else:
                        self.in_group[mod] = gi
            self.bns.append(bns)
            self.convs_out.append(convs)
        self.layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        self.dense_numel = sum(m.weight.numel() for m in self.layers)
        dev = next(model.parameters()).device
        self.mask = [torch.ones(c, dtype=torch.bool, device=dev) for c in self.size]
        # soft gate values (1 except for learned-gate methods); the effective gate is soft * mask
        self.soft = [torch.ones(c, device=dev) for c in self.size]
        self.gates: List[torch.Tensor] = []
        self.gate_grad = [torch.zeros(c, device=dev) for c in self.size]
        self._hooks = []
        self._install()

    # ------------------------------------------------------------------ gating
    def _install(self):
        for gi, bns in enumerate(self.bns):
            for bn, inv in bns:
                self._hooks.append(bn.register_forward_hook(self._make_hook(gi, inv)))

    def _make_hook(self, gi, inv):
        def hook(_m, _i, out):
            g = self.gates[gi]
            return out * g[inv.to(g.device)].view(1, -1, 1, 1)
        return hook

    def refresh_gates(self, learned: List[torch.Tensor] = None, alpha=1.0):
        """Build this step's gate tensors (called before every forward in training).

        The gates are leaf tensors with ``requires_grad`` so that dL/dgate is
        available for channel regrowth; with ``learned`` logits the gate is
        sigmoid(alpha * v) * mask (AutoPruner-style).
        """
        self.gates = []
        for gi in range(len(self.size)):
            if learned is not None:
                g = torch.sigmoid(alpha * learned[gi]) * self.mask[gi]
                self.soft[gi] = g.detach()
            else:
                g = (self.soft[gi] * self.mask[gi]).clone().requires_grad_(True)
            self.gates.append(g)

    @torch.no_grad()
    def accumulate_gate_grad(self):
        for gi, g in enumerate(self.gates):
            if g.is_leaf and g.grad is not None:     # learned (AutoPruner) gates are not leaves
                self.gate_grad[gi] += g.grad.abs()

    def bn_modules(self):
        return {f'g{gi}_bn{j}': bn for gi, bns in enumerate(self.bns) for j, (bn, _) in enumerate(bns)}

    # ------------------------------------------------------------------ parameter accounting
    def params_for(self, kept: List[int]) -> int:
        """Conv/linear weight count of the compact model with ``kept[g]`` channels per group."""
        total = 0
        for m in self.layers:
            gi_o, gi_i = self.out_group.get(m), self.in_group.get(m)
            if isinstance(m, nn.Conv2d):
                cout = kept[gi_o] if gi_o is not None else m.out_channels
                cin = kept[gi_i] if gi_i is not None else m.in_channels
                k = m.kernel_size[0] * m.kernel_size[1]
                if m.groups > 1 and m.groups == m.in_channels:   # depthwise
                    total += k * cout
                else:
                    total += k * cin * cout // m.groups
            else:
                cin = kept[gi_i] if gi_i is not None else m.in_features
                cout = kept[gi_o] if gi_o is not None else m.out_features
                total += cin * cout
        return total

    def kept(self) -> List[int]:
        return [int(m.sum().item()) for m in self.mask]

    def floor(self, gi):
        return max(1, int(math.ceil(self.min_keep * self.size[gi])))

    def uniform_ratio_for(self, target_params) -> float:
        """Uniform per-group removal ratio r that meets ``target_params`` (bisection)."""
        lo, hi = 0.0, 1.0
        for _ in range(40):
            r = (lo + hi) / 2
            kept = [max(self.floor(g), int(round(c * (1 - r)))) for g, c in enumerate(self.size)]
            if self.params_for(kept) > target_params:
                lo = r
            else:
                hi = r
        return hi

    # ------------------------------------------------------------------ selection
    @torch.no_grad()
    def prune_to_params(self, scores: List[torch.Tensor], target_params: int) -> int:
        """Silence the lowest-scoring active channels until params <= target (global ranking).

        Each group keeps at least ``min_keep`` of its channels. Returns the
        number of channels removed.
        """
        cand = []
        for gi, s in enumerate(scores):
            act = self.mask[gi].nonzero().squeeze(1)
            n_removable = act.numel() - self.floor(gi)
            if n_removable <= 0:
                continue
            order = act[torch.sort(s[act].float(), stable=True).indices][:n_removable]
            vals = s[order].float().cpu().tolist()
            cand.extend((v, gi, c) for v, c in zip(vals, order.cpu().tolist()))
        cand.sort(key=lambda x: x[0])
        kept0 = self.kept()

        def params_after(n):
            kept = list(kept0)
            for _, gi, _c in cand[:n]:
                kept[gi] -= 1
            return self.params_for(kept)

        if params_after(0) <= target_params:
            return 0
        lo, hi = 0, len(cand)          # smallest n with params_after(n) <= target
        while lo < hi:
            mid = (lo + hi) // 2
            if params_after(mid) <= target_params:
                hi = mid
            else:
                lo = mid + 1
        for _, gi, c in cand[:lo]:
            self.mask[gi][c] = False
        return lo

    @torch.no_grad()
    def swap(self, scores: List[torch.Tensor], fraction: float) -> int:
        """Channel prune/regrow: drop the lowest-score ``fraction`` of active channels
        (global ranking) and reactivate as many inactive channels with the largest
        accumulated |dL/dgate|. Returns the number of channels swapped."""
        act, inact = [], []
        for gi in range(len(self.size)):
            s, gg, m = scores[gi].float().cpu(), self.gate_grad[gi].cpu(), self.mask[gi].cpu()
            act += [(float(s[c]), gi, c) for c in m.nonzero().squeeze(1).tolist()]
            inact += [(float(gg[c]), gi, c) for c in (~m).nonzero().squeeze(1).tolist()]
        k = min(int(fraction * len(act)), len(inact))
        if k <= 0:
            return 0
        act.sort(key=lambda x: x[0])
        inact.sort(key=lambda x: -x[0])
        kept = self.kept()
        dropped = 0
        for _, gi, c in act:
            if dropped == k:
                break
            if kept[gi] - 1 >= self.floor(gi):
                self.mask[gi][c] = False
                kept[gi] -= 1
                dropped += 1
        for _, gi, c in inact[:dropped]:
            self.mask[gi][c] = True
        return dropped

    def reset_gate_grad(self):
        for g in self.gate_grad:
            g.zero_()

    # ------------------------------------------------------------------ channel scores
    @torch.no_grad()
    def entropy_scores(self, ent: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Mean over the group's BNs of the per-channel output entropy."""
        out = []
        for gi, bns in enumerate(self.bns):
            vals = [ent[f'g{gi}_bn{j}'][self._to_group(inv).to(ent[f'g{gi}_bn{j}'].device)]
                    for j, (_bn, inv) in enumerate(bns) if f'g{gi}_bn{j}' in ent]
            out.append(torch.stack(vals).mean(0) if vals else torch.zeros(self.size[gi], device=self.mask[gi].device))
        return out

    @staticmethod
    def _to_group(inv):
        # inv maps BN channel -> group index; its inverse maps group index -> BN channel
        return torch.argsort(inv)

    @torch.no_grad()
    def bn_gamma_scores(self) -> List[torch.Tensor]:
        out = []
        for gi, bns in enumerate(self.bns):
            vals = [bn.weight.detach().abs()[self._to_group(inv).to(bn.weight.device)] for bn, inv in bns]
            out.append(torch.stack(vals).mean(0))
        return out

    @torch.no_grad()
    def filter_scores(self, kind='fpgm') -> List[torch.Tensor]:
        """FPGM (distance to all other filters; small = near the geometric median) or L1 norm."""
        out = []
        for gi, convs in enumerate(self.convs_out):
            vals = []
            for conv, idx in convs:
                w = conv.weight.detach()[idx.to(conv.weight.device)].flatten(1)
                if kind == 'fpgm':
                    s = torch.cdist(w, w).sum(1)
                else:
                    s = w.abs().sum(1)
                vals.append(s / (s.mean() + 1e-12))
            out.append(torch.stack(vals).mean(0))
        return out

    def slimming_l1(self) -> torch.Tensor:
        return sum(bn.weight.abs().sum() for bns in self.bns for bn, _ in bns)

    # ------------------------------------------------------------------ physical removal
    @torch.no_grad()
    def compact(self) -> nn.Module:
        """Fold gates into BN affine params, remove silenced channels, drop hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        for gi, bns in enumerate(self.bns):
            g = (self.soft[gi] * self.mask[gi]).float()
            for bn, inv in bns:
                gb = g[inv.to(g.device)]
                bn.weight.mul_(gb)
                bn.bias.mul_(gb)
        for gi, (mod, handler) in enumerate(self.roots):
            idxs = (~self.mask[gi]).nonzero().squeeze(1).tolist()
            if idxs:
                pg = self.DG.get_pruning_group(mod, handler, idxs)
                pg.prune()
        self.gates = []
        return self.model
