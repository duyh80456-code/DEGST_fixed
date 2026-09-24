"""One training loop for every method in ``config.METHODS``.

Step order (per mini-batch t):
    forward (entropy hooks record if an update is due within ``entropy_batches`` steps)
    backward                     -> dense gradients, incl. at inactive positions
    accumulate score statistics  (Taylor/Fisher/movement, |dL/dgate|)
    topology update if due       (mask init, prune/regrow, gradual prune, channel selection)
    zero inactive gradients, optimiser step, re-apply masks
"""
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import metrics
from .config import RunConfig
from .criteria import NEEDS_ENTROPY, ScoreState, grad_scores
from .data import NUM_CLASSES, get_loaders
from .entropy import EntropyTracker
from .models import build_model
from .sparse import MaskManager, cosine_fraction, cubic_sparsity, prunable_layers
from .structured import ChannelGroups


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    correct = total = 0
    loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(x)
        loss += F.cross_entropy(out.float(), y, reduction='sum').item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    model.train()
    return 100.0 * correct / total, loss / total


class Run:
    def __init__(self, cfg: RunConfig, device=None, verbose=True):
        self.cfg, self.verbose = cfg, verbose
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.amp = cfg.amp and self.device.type == 'cuda'
        seed_everything(cfg.seed)
        self.train_loader, self.eval_loader, self.eval_split = get_loaders(
            cfg.dataset, cfg.data_root, cfg.batch_size, cfg.num_workers, cfg.val_size, cfg.seed)
        self.model = build_model(cfg.model, NUM_CLASSES[cfg.dataset]).to(self.device)
        self.dense_weights = sum(m.weight.numel() for m in prunable_layers(self.model).values())
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = cfg.epochs * self.steps_per_epoch
        self.t_warm = cfg.warmup_epochs * self.steps_per_epoch
        self.t_end = int(cfg.end_epoch * self.steps_per_epoch)
        self.t_prune = int(cfg.prune_epoch * self.steps_per_epoch)   # posttrain only
        self.init_scores = None
        self.mm = None            # MaskManager (unstructured)
        self.scores = None        # ScoreState
        self.cg = None            # ChannelGroups (structured)
        self.ap_logits = None     # AutoPruner-style gate logits
        self.tracker = None
        self.update_seconds = 0.0
        self.entropy_seconds = 0.0   # from trackers already removed
        self.log = []
        self._setup()

    # ------------------------------------------------------------------ setup
    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _optimizer(self):
        params = list(self.model.parameters())
        groups = [{'params': params, 'weight_decay': self.cfg.weight_decay}]
        if self.ap_logits is not None:
            groups.append({'params': self.ap_logits, 'weight_decay': 0.0})
        self.opt = torch.optim.SGD(groups, lr=self.cfg.lr, momentum=self.cfg.momentum)

    def _setup(self):
        c = self.cfg
        if c.kind in ('dst', 'gradual', 'oneshot_init', 'posttrain'):
            self._init_unstructured()
            if c.kind == 'posttrain':
                # initial weights, for the lottery-ticket rewind after pruning
                self.init_weights = {n: m.weight.detach().clone() for n, m in self.mm.layers.items()}
            if c.init == 'random' and c.kind == 'dst':
                g = torch.Generator().manual_seed(c.seed)
                self.mm.init_random(c.sparsity, generator=g)
        if c.kind in ('structured', 'hybrid'):
            ex = torch.zeros(1, 3, 32, 32, device=self.device)
            self.cg = ChannelGroups(self.model, ex, self.model.linear, c.min_channel_keep)
            share = c.hybrid_struct_share if c.kind == 'hybrid' else 1.0
            self.struct_sparsity = share * c.sparsity
            if c.channel_criterion == 'entropy':
                self.tracker = EntropyTracker(self.cg.bn_modules(), 'output', c.bins, c.entropy_reduction,
                                              c.entropy_max_samples)
            if c.channel_criterion == 'learned':
                keep = 1 - self.cg.uniform_ratio_for((1 - self.struct_sparsity) * self.dense_weights)
                self.ap_keep = keep
                self.ap_logits = [torch.full((s,), 1.0, device=self.device, requires_grad=True)
                                  for s in self.cg.size]
            self.struct_end = self.t_end if c.kind == 'structured' else (self.t_warm + self.t_end) // 2
            # hybrid: unstructured mask is initialised entropy_batches steps after compaction
            self.hybrid_init = self.struct_end + c.entropy_batches
        self._optimizer()

    def _init_unstructured(self):
        c = self.cfg
        self.mm = MaskManager(self.model, c.allocation, c.min_density)
        self.scores = ScoreState(self.mm.layers, c.criterion)
        if c.criterion in NEEDS_ENTROPY and self.tracker is None:
            self.tracker = EntropyTracker(self.mm.layers, 'input', c.bins, c.entropy_reduction,
                                          c.entropy_max_samples)

    # ------------------------------------------------------------------ schedule helpers
    def lr_at(self, t):
        start, length = 0, self.total_steps
        if self.cfg.kind == 'posttrain':    # separate cosine cycles for dense training and fine-tuning
            if t < self.t_prune:
                length = self.t_prune
            else:
                start, length = self.t_prune, self.total_steps - self.t_prune
        return 0.5 * self.cfg.lr * (1 + math.cos(math.pi * (t - start) / max(1, length)))

    def _due(self, t, start, end):
        return start <= t <= end and t % self.cfg.update_interval == 0

    def _fraction(self, t, start):
        if self.cfg.ratio_decay == 'constant':
            return self.cfg.prune_ratio if t < self.t_end else 0.0
        return cosine_fraction(t, start, self.t_end, self.cfg.prune_ratio)

    def _mask_init_step(self):
        c = self.cfg
        if c.kind == 'dst' and c.init == 'score':
            return self.t_warm
        if c.kind == 'oneshot_init':
            return 0
        if c.kind == 'hybrid':
            return self.hybrid_init
        return None

    def _entropy_active(self, t):
        if self.tracker is None:
            return False
        nb = self.cfg.entropy_batches
        init = self._mask_init_step()
        if init is not None and 0 <= init - t < nb:
            return True
        return (-t) % self.cfg.update_interval < nb

    # ------------------------------------------------------------------ topology updates
    def _unstructured_update(self, t):
        c, mm = self.cfg, self.mm
        init = self._mask_init_step()
        if mm is None:
            return
        if init is not None and t == init:
            ent = self.tracker.reduce() if self.tracker else None
            sp = c.sparsity if c.kind != 'hybrid' else self.hybrid_inner_sparsity
            mm.init_from_scores(self.scores.scores(ent), sp)
            self.scores.reset()
            self._log(f'  [t={t}] mask init -> sparsity {mm.sparsity(self.dense_weights):.4f}')
            return
        if c.kind == 'posttrain':
            self._posttrain_update(t)
            return
        if c.kind == 'dst':
            start = init if init is not None else 0
            if t <= start or not self._due(t, start + 1, self.t_end):
                return
            frac = self._fraction(t, start)
            ent = self.tracker.reduce() if self.tracker else None
            info = mm.prune_regrow(self.scores.scores(ent), grad_scores(mm.layers), frac)
            mm.reset_momentum(self.opt, info['changed'])
            self.scores.reset()
            self.log.append({'t': t, 'swap_fraction': frac, 'pruned': info['pruned'], 'grown': info['grown']})
        elif c.kind == 'gradual' and (self._due(t, self.t_warm, self.t_end) or t == self.t_end):
            s = cubic_sparsity(t, self.t_warm, self.t_end, c.sparsity)
            removed = mm.prune_to_sparsity(self.scores.scores(), s)
            mm.reset_momentum(self.opt, removed)
            self.scores.reset()
        elif c.kind == 'hybrid' and init is not None and t > init and self._due(t, init + 1, self.t_end):
            frac = self._fraction(t, init)
            ent = self.tracker.reduce() if self.tracker else None
            info = mm.prune_regrow(self.scores.scores(ent), grad_scores(mm.layers), frac)
            mm.reset_momentum(self.opt, info['changed'])
            self.scores.reset()

    @torch.no_grad()
    def _posttrain_update(self, t):
        """One-shot pruning after dense training, optional rewind (pre-revision pipeline)."""
        c, mm = self.cfg, self.mm
        if t == 0 and c.score_at == 'init':
            self.init_scores = {n: s.clone() for n, s in self.scores.scores().items()}
        if t != self.t_prune:
            return
        scores = self.init_scores if c.score_at == 'init' else self.scores.scores()
        mm.init_from_scores(scores, c.sparsity)
        if c.rewind:
            for n, m in mm.layers.items():
                m.weight.copy_(self.init_weights[n] * mm.masks[n])
        # fresh optimiser state for fine-tuning; drop this step's gradients of the dense model
        for p in self.model.parameters():
            p.grad = None
        self._optimizer()
        self._log(f'  [t={t}] one-shot prune ({c.criterion}, rewind={c.rewind}) -> '
                  f'sparsity {mm.sparsity(self.dense_weights):.4f}')

    def _channel_scores(self):
        cc, cg = self.cfg.channel_criterion, self.cg
        if cc == 'entropy':
            return cg.entropy_scores(self.tracker.reduce())
        if cc == 'bn_gamma':
            return cg.bn_gamma_scores()
        if cc in ('fpgm', 'l1'):
            return cg.filter_scores(cc)
        if cc == 'learned':
            return [s.detach() for s in cg.soft]
        raise ValueError(cc)

    def _structured_update(self, t):
        c, cg = self.cfg, self.cg
        if cg is None or t > self.struct_end or t < self.t_warm:
            return
        final = t == self.struct_end
        target_s = None
        if c.channel_schedule == 'gradual' and (self._due(t, self.t_warm, self.struct_end) or final):
            target_s = cubic_sparsity(t, self.t_warm, self.struct_end, self.struct_sparsity)
        elif c.channel_schedule == 'oneshot' and final:
            target_s = self.struct_sparsity
        if target_s is None:
            return
        scores = self._channel_scores()
        swapped = 0
        if c.channel_swap and not final and t > self.t_warm:
            swapped = cg.swap(scores, cosine_fraction(t, self.t_warm, self.struct_end, c.prune_ratio))
        removed = cg.prune_to_params(scores, int((1 - target_s) * self.dense_weights))
        cg.reset_gate_grad()
        self.log.append({'t': t, 'channels_kept': sum(cg.kept()), 'swapped': swapped, 'removed': removed,
                         'weights': cg.params_for(cg.kept())})
        if final:
            self._compact(t)

    def _compact(self, t):
        cg = self.cg
        self.model = cg.compact()
        self.cg = None
        self.ap_logits = None
        kept_w = sum(m.weight.numel() for m in prunable_layers(self.model).values())
        self._log(f'  [t={t}] compacted: {kept_w:,} / {self.dense_weights:,} weights '
                  f'({1 - kept_w / self.dense_weights:.4f} structured sparsity)')
        if self.cfg.kind == 'hybrid':
            self.hybrid_inner_sparsity = max(0.0, 1 - (1 - self.cfg.sparsity) * self.dense_weights / kept_w)
            if self.tracker is not None:
                self.entropy_seconds += self.tracker.seconds
                self.tracker.remove()
            self.tracker = None
            self._init_unstructured()
            self.mm.dense_numel = kept_w
        elif self.tracker is not None:
            self.entropy_seconds += self.tracker.seconds
            self.tracker.remove()
            self.tracker = None
        self._optimizer()

    # ------------------------------------------------------------------ main loop
    def train(self):
        c, dev = self.cfg, self.device
        history, t = [], 0
        _sync(dev)
        t_train0 = time.perf_counter()
        epoch_times = []
        for epoch in range(c.epochs):
            self.model.train()
            _sync(dev)
            te0 = time.perf_counter()
            run_loss = run_correct = run_n = 0
            for x, y in self.train_loader:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                for g in self.opt.param_groups:
                    g['lr'] = self.lr_at(t)
                if self.cg is not None:
                    if self.ap_logits is not None and t >= self.t_warm:
                        prog = min(1.0, (t - self.t_warm) / max(1, self.struct_end - self.t_warm))
                        self.cg.refresh_gates(self.ap_logits, 1 + (c.autopruner_alpha_max - 1) * prog)
                    else:
                        self.cg.refresh_gates()
                if self.tracker is not None:
                    self.tracker.active = self._entropy_active(t)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=self.amp):
                    out = self.model(x)
                    loss = F.cross_entropy(out.float(), y)
                    total = loss
                    if self.cg is not None and c.slimming_lambda > 0:
                        total = total + c.slimming_lambda * self.cg.slimming_l1()
                    if self.cg is not None and self.ap_logits is not None and t >= self.t_warm:
                        reg = sum((g.mean() - self.ap_keep) ** 2 for g in self.cg.gates)
                        total = total + c.autopruner_lambda * reg
                self.opt.zero_grad(set_to_none=True)
                total.backward()

                _sync(dev)
                tu0 = time.perf_counter()
                if self.scores is not None:
                    self.scores.accumulate()
                if self.cg is not None:
                    self.cg.accumulate_gate_grad()
                self._structured_update(t)
                self._unstructured_update(t)
                if self.mm is not None:
                    self.mm.mask_grads()
                _sync(dev)
                self.update_seconds += time.perf_counter() - tu0

                self.opt.step()
                if self.mm is not None:
                    self.mm.apply()
                run_loss += loss.item() * y.numel()
                run_correct += (out.argmax(1) == y).sum().item()
                run_n += y.numel()
                t += 1
            _sync(dev)
            epoch_times.append(time.perf_counter() - te0)
            acc, eval_loss = evaluate(self.model, self.eval_loader, dev, self.amp)
            rec = {'epoch': epoch + 1, 'train_loss': run_loss / run_n, 'train_acc': 100 * run_correct / run_n,
                   f'{self.eval_split}_acc': acc, f'{self.eval_split}_loss': eval_loss,
                   'epoch_seconds': epoch_times[-1]}
            if self.mm is not None:
                rec['sparsity'] = self.mm.sparsity(self.dense_weights) if c.kind != 'hybrid' or self.cg is None \
                    else None
            history.append(rec)
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == c.epochs:
                self._log(f'epoch {epoch + 1}/{c.epochs} loss {rec["train_loss"]:.3f} '
                          f'train {rec["train_acc"]:.2f} {self.eval_split} {acc:.2f} ({epoch_times[-1]:.1f}s)')
        _sync(dev)
        train_seconds = time.perf_counter() - t_train0
        entropy_seconds = self.entropy_seconds + (self.tracker.seconds if self.tracker else 0.0)
        if self.tracker is not None:
            self.tracker.remove()
        return history, {'train_seconds': train_seconds, 'gpu_hours': train_seconds / 3600,
                         'epoch_seconds_mean': float(np.mean(epoch_times)),
                         'topology_update_seconds': self.update_seconds,
                         'entropy_hook_seconds': entropy_seconds}

    # ------------------------------------------------------------------ final measurement
    def measure(self):
        c, m = self.cfg, self.model
        m.eval()
        pc = metrics.param_counts(m)
        fl = metrics.flops(m)
        dense_model = build_model(c.model, NUM_CLASSES[c.dataset]).to(self.device)
        fl0 = metrics.flops(dense_model)
        res = {
            **pc, **fl,
            'dense_weights': self.dense_weights,
            'dense_params_total': metrics.param_counts(dense_model)['params_total'],
            'dense_flops': fl0['flops_dense'],
            'sparsity_actual': 1 - pc['weights_nonzero'] / self.dense_weights,
            'flops_reduction_sparse': 1 - fl['flops_sparse'] / fl0['flops_dense'],
            'flops_reduction_dense_arch': 1 - fl['flops_dense'] / fl0['flops_dense'],
            'speed': metrics.measure_speed(m, c.speed_batches, self.device, c.speed_warmup, c.speed_iters),
            'speed_protocol': metrics.protocol(self.device, c.speed_warmup, c.speed_iters, c.speed_batches),
        }
        return res


def run(cfg: RunConfig, device=None, verbose=True) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    r = Run(cfg, device, verbose)
    r._log(f'== {cfg.run_id()} on {r.device} ({r.steps_per_epoch} it/epoch)')
    history, cost = r.train()
    final = r.measure()
    key = f'{r.eval_split}_acc'
    result = {
        'run_id': cfg.run_id(), 'config': cfg.to_dict(), 'eval_split': r.eval_split,
        'final_acc': history[-1][key],          # final-epoch accuracy; never the max over the eval set
        'history': history, 'topology_log': r.log[-200:], 'cost': cost, 'final': final,
    }
    path = os.path.join(cfg.out_dir, cfg.run_id() + '.json')
    with open(path + '.tmp', 'w') as f:
        json.dump(result, f, indent=1)
    os.replace(path + '.tmp', path)
    if cfg.save_checkpoint:
        os.makedirs(os.path.join(cfg.out_dir, 'checkpoints'), exist_ok=True)
        torch.save(r.model.cpu(), os.path.join(cfg.out_dir, 'checkpoints', cfg.run_id() + '.pt'))
    r._log(f'== done {cfg.run_id()}: {key} {result["final_acc"]:.2f}  sparsity {final["sparsity_actual"]:.4f}  '
           f'FLOPs red. sparse {final["flops_reduction_sparse"]:.3f} / dense-arch {final["flops_reduction_dense_arch"]:.3f}')
    return result
