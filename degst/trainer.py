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
from .data import IMAGE_SIZE, NUM_CLASSES, get_loaders
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


class _SectionTimer:
    """Time spent in a code section that runs every step, without a host sync per step.

    On CUDA each section is bracketed by two events and the elapsed times are read
    once per epoch (``flush``, after the epoch's final sync): this is the stream
    time between the section's start and end, including any idle gap while the host
    runs the section's Python code. On CPU it is plain wall-clock time.
    """

    def __init__(self, device):
        self.cuda = device.type == 'cuda'
        self.pairs, self.cpu_seconds, self._t0 = [], 0.0, None

    def start(self):
        if self.cuda:
            self.pairs.append((torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)))
            self.pairs[-1][0].record()
        else:
            self._t0 = time.perf_counter()

    def stop(self):
        if self.cuda:
            self.pairs[-1][1].record()
        else:
            self.cpu_seconds += time.perf_counter() - self._t0

    def flush(self) -> float:
        """Seconds since the last flush; on CUDA call after a device sync."""
        s = sum(a.elapsed_time(b) for a, b in self.pairs) / 1000 if self.cuda else self.cpu_seconds
        self.pairs, self.cpu_seconds = [], 0.0
        return s


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
        if self.device.type == 'cuda':
            # fixed input shapes: let cuDNN pick the fastest conv algorithms (runs are not bitwise
            # deterministic anyway: no deterministic mode is set)
            torch.backends.cudnn.benchmark = True
        self.update_timer = _SectionTimer(self.device)
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
            hw = IMAGE_SIZE[c.dataset]
            ex = torch.zeros(1, 3, hw, hw, device=self.device)
            self.cg = ChannelGroups(self.model, ex, self.model.linear, c.min_channel_keep)
            share = c.hybrid_struct_share if c.kind == 'hybrid' else 1.0
            self.struct_sparsity = share * c.sparsity
            if c.channel_criterion == 'entropy':
                self.tracker = EntropyTracker(self.cg.bn_modules(), 'output', c.bins, c.entropy_reduction,
                                              c.entropy_max_samples, estimator=c.entropy_estimator)
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
        self.scores = ScoreState(self.mm.layers, c.criterion, seed=c.seed)
        if c.criterion in NEEDS_ENTROPY and self.tracker is None:
            self.tracker = EntropyTracker(self.mm.layers, 'input', c.bins, c.entropy_reduction,
                                          c.entropy_max_samples, estimator=c.entropy_estimator)

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

    def _save_mid_checkpoint(self, epoch, t):
        """Model and current masks at the end of ``epoch`` (for analysis/score_overlap.py)."""
        d = os.path.join(self.cfg.out_dir, 'checkpoints')
        os.makedirs(d, exist_ok=True)
        c = self.cfg
        masks = {n: m.detach().cpu().clone() for n, m in self.mm.masks.items()} if self.mm is not None else None
        # what the loop would do at the next update, so the analysis can replay that exact step
        init = self._mask_init_step()
        frac = self._fraction(t, init if init is not None else 0) if c.kind == 'dst' else None
        path = os.path.join(d, f'{c.run_id()}_ep{epoch}.pt')
        torch.save({'model': self.model, 'masks': masks, 'epoch': epoch, 'step': t, 'swap_fraction': frac,
                    'update_interval': c.update_interval, 'entropy_batches': c.entropy_batches,
                    'seed': c.seed}, path + '.tmp')
        os.replace(path + '.tmp', path)
        self._log(f'  saved {path}')

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

                self.update_timer.start()
                if self.scores is not None:
                    self.scores.accumulate()
                if self.cg is not None:
                    self.cg.accumulate_gate_grad()
                self._structured_update(t)
                self._unstructured_update(t)
                if self.mm is not None:
                    self.mm.mask_grads()
                self.update_timer.stop()

                self.opt.step()
                if self.mm is not None:
                    self.mm.apply()
                # running sums stay on the device: no host sync per step
                run_loss += loss.detach().double() * y.numel()
                run_correct += (out.argmax(1) == y).sum()
                run_n += y.numel()
                t += 1
            _sync(dev)
            epoch_times.append(time.perf_counter() - te0)
            self.update_seconds += self.update_timer.flush()
            run_loss, run_correct = float(run_loss), int(run_correct)
            # the test set is only read every eval_every epochs; the reported accuracy is the last epoch's
            last = epoch + 1 == c.epochs
            if epoch == 0 or last or (epoch + 1) % max(1, c.eval_every) == 0:
                acc, eval_loss = evaluate(self.model, self.eval_loader, dev, self.amp)
            else:
                acc = eval_loss = None
            rec = {'epoch': epoch + 1, 'train_loss': run_loss / run_n, 'train_acc': 100 * run_correct / run_n,
                   f'{self.eval_split}_acc': acc, f'{self.eval_split}_loss': eval_loss,
                   'epoch_seconds': epoch_times[-1]}
            if self.mm is not None:
                rec['sparsity'] = self.mm.sparsity(self.dense_weights) if c.kind != 'hybrid' or self.cg is None \
                    else None
            history.append(rec)
            if epoch + 1 in c.checkpoint_epochs and epoch + 1 < c.epochs:
                self._save_mid_checkpoint(epoch + 1, t)
            if epoch == 0 or (epoch + 1) % 10 == 0 or last:
                acc_s = f'{acc:.2f}' if acc is not None else '--'
                self._log(f'epoch {epoch + 1}/{c.epochs} loss {rec["train_loss"]:.3f} '
                          f'train {rec["train_acc"]:.2f} {self.eval_split} {acc_s} ({epoch_times[-1]:.1f}s)')
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
        hw = IMAGE_SIZE[c.dataset]
        fl = metrics.flops(m, (1, 3, hw, hw))
        dense_model = build_model(c.model, NUM_CLASSES[c.dataset]).to(self.device)
        fl0 = metrics.flops(dense_model, (1, 3, hw, hw))
        res = {
            **pc, **fl,
            'dense_weights': self.dense_weights,
            'dense_params_total': metrics.param_counts(dense_model)['params_total'],
            'dense_flops': fl0['flops_dense'],
            'sparsity_actual': 1 - pc['weights_nonzero'] / self.dense_weights,
            'flops_reduction_sparse': 1 - fl['flops_sparse'] / fl0['flops_dense'],
            'flops_reduction_dense_arch': 1 - fl['flops_dense'] / fl0['flops_dense'],
            'speed': metrics.measure_speed(m, c.speed_batches, self.device, c.speed_warmup, c.speed_iters, hw),
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
    # checkpoint first and both files atomically: the JSON marks a run as done (--skip-existing), so it
    # must only appear once the checkpoint is complete
    if cfg.save_checkpoint:
        os.makedirs(os.path.join(cfg.out_dir, 'checkpoints'), exist_ok=True)
        ck = os.path.join(cfg.out_dir, 'checkpoints', cfg.run_id() + '.pt')
        torch.save(r.model.cpu(), ck + '.tmp')
        os.replace(ck + '.tmp', ck)
    path = os.path.join(cfg.out_dir, cfg.run_id() + '.json')
    with open(path + '.tmp', 'w') as f:
        json.dump(result, f, indent=1)
    os.replace(path + '.tmp', path)
    r._log(f'== done {cfg.run_id()}: {key} {result["final_acc"]:.2f}  sparsity {final["sparsity_actual"]:.4f}  '
           f'FLOPs red. sparse {final["flops_reduction_sparse"]:.3f} / dense-arch {final["flops_reduction_dense_arch"]:.3f}')
    return result
