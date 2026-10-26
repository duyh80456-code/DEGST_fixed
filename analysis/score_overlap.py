"""Which weights would each drop criterion remove? Diagnostics on saved checkpoints.

    python analysis/score_overlap.py results/checkpoints/*_ep*.pt results/midckpt/checkpoints/*_ep*.pt
    python analysis/score_overlap.py A.pt --compare C.pt          # + per-layer density of both final masks

On the model state of each checkpoint, the next global prune step of the DST loop
is replayed with every in-loop criterion (|w|, H, H|w|, H_robust|w|, Taylor,
Fisher, |g|, random), using the same ScoreState / MaskManager code as training.
Mid-training checkpoints carry the loop's state, so the replay uses the loop's own
numbers: the swap fraction of that step, entropy from ``entropy_batches`` batches,
and Taylor/Fisher/|g| averaged over ``update_interval`` batches (as between two
updates). Final checkpoints fall back to 20 % / 4 / 400 and mask = weight != 0.

Reported:
  * per layer: input-channel entropy (mean, CV), share of exact zeros in the input
    (post-ReLU sparsity), Spearman correlation of H with that share, and how many
    weights each criterion drops there (allocation between layers);
  * Jaccard overlap of the drop sets for every pair of criteria;
  * Jaccard with the |w| drop set *within* each layer at |w|'s per-layer count
    (ranking effect only, allocation held fixed);
  * FLOPs of the dropped weights, as % of the active FLOPs.
Writes results/analysis/overlap_<checkpoint name>.json per checkpoint.
"""
import argparse
import json
import os
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from degst.criteria import ScoreState  # noqa: E402
from degst.data import get_loaders  # noqa: E402
from degst.entropy import channel_energy, channel_entropy, channel_entropy_robust  # noqa: E402
from degst.sparse import MaskManager, _bottom_k, prunable_layers  # noqa: E402

CRITERIA = ('magnitude', 'entropy', 'entropy_magnitude', 'entropy_magnitude_robust', 'taylor', 'fisher',
            'gradient', 'random')
# phase 2: S1 (Wanda's score) and S3 (magnitude-anchored composite) at every beta of the sweep;
# composite names are 'composite_<signal>_b<beta>', e.g. 'composite_energy_b0.5'
PHASE2_CRITERIA = ('energy_magnitude',) + tuple(f'composite_energy_b{b:g}' for b in (0.1, 0.25, 0.5, 1.0, 2.0))
SHORT = {'magnitude': '|w|', 'entropy': 'H', 'entropy_magnitude': 'H|w|', 'entropy_magnitude_robust': 'Hr|w|',
         'taylor': 'Taylor', 'fisher': 'Fisher', 'gradient': '|g|', 'random': 'rand', 'energy_magnitude': 'E|w|'}


def short(c):
    m = re.match(r'composite_(\w+?)_b([\d.]+)$', c)
    return f'C{m.group(1)[0]}{m.group(2)}' if m else SHORT[c]


def make_state(layers, c, seed):
    """ScoreState for a criterion name of this script (robust entropy and composite names included)."""
    m = re.match(r'composite_(\w+?)_b([\d.]+)$', c)
    if m:
        return ScoreState(layers, 'composite', seed=seed, beta=float(m.group(2)), signal=m.group(1))
    return ScoreState(layers, 'entropy_magnitude' if c == 'entropy_magnitude_robust' else c, seed=seed)
REFERENCE = 'magnitude'
DEFAULTS = {'swap_fraction': 0.2, 'entropy_batches': 4, 'update_interval': 400}


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 and b.numel() == 0:
        return 1.0
    inter = int(torch.isin(a, b).sum().item())
    return inter / (a.numel() + b.numel() - inter)


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float('nan')
    rx = torch.argsort(torch.argsort(x)).double()
    ry = torch.argsort(torch.argsort(y)).double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = rx.norm() * ry.norm()
    return float((rx * ry).sum() / den) if den > 0 else float('nan')


def collect(model: nn.Module, batches, device, criteria=CRITERIA, bins=30, amp=True, entropy_batches=4, seed=0):
    """Scores of every criterion on the current weights, without an optimiser step.

    ``batches`` yields (x, y). Entropy (plain and robust) and the share of exact zeros are recorded
    in the forward pass of the first ``entropy_batches`` batches; gradient statistics are
    accumulated after the backward pass of every batch.
    Returns (scores {criterion: {layer: tensor}}, channel stats {'H', 'H_robust', 'zero'}: {layer: [C_in]},
    positions {layer: int}).
    """
    layers = prunable_layers(model)
    bufs = {k: {n: [] for n in layers} for k in ('H', 'H_robust', 'zero', 'energy')}
    positions, handles, rec = {}, [], {'on': True}

    def pre(n):
        def hook(_m, inp):
            if rec['on']:
                x = inp[0]
                bufs['H'][n].append(channel_entropy(x, bins))
                bufs['H_robust'][n].append(channel_entropy_robust(x, bins))
                bufs['energy'][n].append(channel_energy(x))
                bufs['zero'][n].append((x.detach() == 0).float().transpose(0, 1).reshape(x.shape[1], -1).mean(1))
        return hook

    for n, m in layers.items():
        handles.append(m.register_forward_pre_hook(pre(n)))
        handles.append(m.register_forward_hook(
            lambda _m, _i, out, n=n: positions.__setitem__(n, out.shape[2] * out.shape[3] if out.dim() == 4 else 1)))
    states = {c: make_state(layers, c, seed) for c in criteria}
    model.train()
    dev = torch.device(device)
    for i, (x, y) in enumerate(batches):
        rec['on'] = i < entropy_batches
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        model.zero_grad(set_to_none=True)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp and dev.type == 'cuda'):
            loss = F.cross_entropy(model(x).float(), y)
        loss.backward()
        for s in states.values():
            s.accumulate()
    for h in handles:
        h.remove()
    stats = {k: {n: torch.stack(v).mean(0) for n, v in b.items() if v} for k, b in bufs.items()}

    def stat_for(c, s):
        if s.activation_stat == 'energy':
            return stats['energy']
        return stats['H_robust'] if c == 'entropy_magnitude_robust' else stats['H']

    scores = {c: s.scores(stat_for(c, s)) for c, s in states.items()}
    model.zero_grad(set_to_none=True)
    return scores, stats, positions


@torch.no_grad()
def compare_drop_sets(model: nn.Module, scores, stats=None, masks=None, fraction=0.2, min_density=0.01,
                      positions=None, reference=REFERENCE):
    """Simulate one global prune step with each criterion's scores on the same mask.

    ``stats`` is the channel-statistics dict from ``collect`` (or {'H': entropy}); optional.
    """
    mm = MaskManager(model, 'global', min_density)
    base = {n: (masks[n].to(m.weight.device).bool() if masks is not None else m.weight != 0)
            for n, m in mm.layers.items()}
    mm.masks = {n: v.clone() for n, v in base.items()}
    k = int(fraction * mm.nnz())
    crits = list(scores)
    drops = {}
    for c in crits:
        mm.masks = {n: v.clone() for n, v in base.items()}
        drops[c] = mm._prune_global(scores[c], k)
    mm.masks = base
    positions, stats = positions or {}, stats or {}

    rows, flops = [], {c: 0 for c in crits}
    flat = {c: [] for c in crits}
    pooled_h, pooled_z = [], []
    for i, (n, m) in enumerate(mm.layers.items()):
        act = base[n].view(-1).nonzero().squeeze(1)
        empty = act.new_empty(0)
        d = {c: drops[c].get(n, empty) for c in crits}
        pos = positions.get(n, 1)
        for c in crits:
            flops[c] += 2 * pos * d[c].numel()
            flat[c].append(d[c] + i * (1 << 40))
        # ranking effect: each criterion's bottom-k inside this layer at the reference's count
        k_ref = d[reference].numel()
        ref_sc = scores[reference][n].reshape(-1)[act].float()
        j_in, rho = {}, {}
        for c in crits:
            sc = scores[c][n].reshape(-1)[act].float()
            within = act[_bottom_k(sc, k_ref)] if k_ref else empty
            j_in[c] = _jaccard(d[reference], within)
            rho[c] = _spearman(ref_sc, sc)
        row = {'layer': n, 'numel': m.weight.numel(), 'active': act.numel(),
               'density': act.numel() / m.weight.numel(), 'positions': pos,
               'drop': {c: d[c].numel() for c in crits}, 'jaccard_within': j_in, 'spearman': rho}
        h = stats.get('H', {}).get(n)
        if h is not None:
            h = h.float().cpu()
            row.update(H_mean=float(h.mean()),
                       H_cv=float(h.std() / h.mean().clamp_min(1e-12)) if h.numel() > 1 else 0.0,
                       dead_share=float((h < 1e-3).float().mean()))
            z = stats.get('zero', {}).get(n)
            if z is not None:
                z = z.float().cpu()
                row.update(zero_share=float(z.mean()), spearman_H_zero=_spearman(h, z))
                if i > 0:          # the first layer reads the image: no ReLU zeros
                    pooled_h.append(h)
                    pooled_z.append(z)
            hr = stats.get('H_robust', {}).get(n)
            if hr is not None:
                row['spearman_H_Hrobust'] = _spearman(h, hr.float().cpu())
        rows.append(row)

    cat = {c: torch.cat(flat[c]) for c in crits}
    k_ref = sum(r['drop'][reference] for r in rows)
    active_flops = sum(2 * r['positions'] * r['active'] for r in rows)
    summary = {
        'k_dropped': k, 'fraction': fraction, 'reference': reference,
        'jaccard': {a: {b: _jaccard(cat[a], cat[b]) for b in crits} for a in crits},
        'jaccard_within_vs_ref': {c: sum(r['jaccard_within'][c] * r['drop'][reference] for r in rows) / max(1, k_ref)
                                  for c in crits},
        'flops_dropped_pct': {c: 100 * flops[c] / max(1, active_flops) for c in crits},
    }
    if pooled_h:
        summary['spearman_H_zero_pooled'] = _spearman(torch.cat(pooled_h), torch.cat(pooled_z))
    return rows, summary


def load_checkpoint(path, device):
    """(model, masks, loop state); plain final checkpoints have no masks and no loop state."""
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and 'model' in obj:
        state = {k: obj.get(k) for k in ('epoch', 'swap_fraction', 'entropy_batches', 'update_interval', 'seed')}
        return obj['model'], obj.get('masks'), state
    return obj, None, {}


def parse_name(name):
    """cifar10_resnet18_degst_magnitude_s90_seed0[_hash][_ep50] -> dataset, model, method, sparsity %, seed, epoch."""
    m = re.match(r'^([^_]+)_([^_]+)_(.+)_s(\d+)_seed(\d+)(?:_[0-9a-f]{8})?(?:_ep(\d+))?$', name)
    if not m:
        return {'dataset': name.split('_')[0]}
    return {'dataset': m.group(1), 'model': m.group(2), 'method': m.group(3), 'sparsity_pct': int(m.group(4)),
            'seed': int(m.group(5)), 'epoch': int(m.group(6)) if m.group(6) else None}


def take(loader, n):
    """The first ``n`` batches, cycling over the loader if it is shorter."""
    i = 0
    while i < n:
        for b in loader:
            if i >= n:
                return
            yield b
            i += 1


def _print(name, rows, summary, crits):
    print(f'== {name}: replay dropping {summary["k_dropped"]} weights ({summary["fraction"]:.1%} of active)')
    print(f'{"layer":24s} {"dens":>5s} {"H":>5s} {"H cv":>5s} {"zero":>5s} {"rhoHz":>6s} ' +
          ' '.join(f'{short(c):>7s}' for c in crits))
    nan = float('nan')
    for r in rows:
        print(f'{r["layer"]:24s} {r["density"]:5.3f} {r.get("H_mean", nan):5.2f} {r.get("H_cv", nan):5.2f} '
              f'{r.get("zero_share", nan):5.2f} {r.get("spearman_H_zero", nan):6.2f} ' +
              ' '.join(f'{r["drop"][c]:7d}' for c in crits))
    print('Jaccard of drop sets (global):')
    print(f'{"":8s}' + ''.join(f'{short(c):>8s}' for c in crits))
    for a in crits:
        print(f'{short(a):8s}' + ''.join(f'{summary["jaccard"][a][b]:8.3f}' for b in crits))
    print('within layers vs |w| : ' + ', '.join(f'{short(c)} {v:.3f}' for c, v in summary['jaccard_within_vs_ref'].items()))
    print('FLOPs dropped (%)    : ' + ', '.join(f'{short(c)} {v:.2f}' for c, v in summary['flops_dropped_pct'].items()))
    if 'spearman_H_zero_pooled' in summary:
        print(f'Spearman(H, share of zeros) over all channels after the first layer: '
              f'{summary["spearman_H_zero_pooled"]:+.2f}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoints', nargs='+')
    p.add_argument('--compare', nargs='*', default=[], help='print per-layer density of these next to the first')
    p.add_argument('--criteria', default=','.join(CRITERIA))
    p.add_argument('--phase2', action='store_true',
                   help='also replay S1 (energy_magnitude) and S3 (composite) at every beta of the sweep')
    p.add_argument('--dataset', default=None, help='default: taken from the checkpoint name')
    p.add_argument('--data-root', default='./data')
    p.add_argument('--fraction', type=float, default=None, help='default: the loop\'s swap fraction at that step')
    p.add_argument('--batches', type=int, default=None, help='entropy batches (default: entropy_batches)')
    p.add_argument('--grad-batches', type=int, default=None,
                   help='batches for Taylor/Fisher/|g| (default: update_interval)')
    p.add_argument('--bins', type=int, default=30)
    p.add_argument('--min-density', type=float, default=0.01)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default=os.path.join('results', 'analysis'))
    a = p.parse_args()
    crits = [c for c in a.criteria.split(',') if c] + (list(PHASE2_CRITERIA) if a.phase2 else [])
    if REFERENCE not in crits:
        crits.insert(0, REFERENCE)
    os.makedirs(a.out_dir, exist_ok=True)
    loaders = {}

    for ck in a.checkpoints:
        name = os.path.splitext(os.path.basename(ck))[0]
        meta = parse_name(name)
        dataset = a.dataset or meta['dataset']
        if dataset not in loaders:
            loaders[dataset] = get_loaders(dataset, a.data_root, 128, a.num_workers, seed=a.seed)[0]
        torch.manual_seed(a.seed)
        model, masks, state = load_checkpoint(ck, a.device)

        def pick(arg, key):
            return arg if arg is not None else (state.get(key) if state.get(key) is not None else DEFAULTS[key])

        fraction = pick(a.fraction, 'swap_fraction')
        ent_b, grad_b = pick(a.batches, 'entropy_batches'), pick(a.grad_batches, 'update_interval')
        if fraction <= 0:
            print(f'skip {name}: the loop makes no topology change at this step (fraction 0)')
            continue
        scores, stats, positions = collect(model, take(loaders[dataset], max(ent_b, grad_b)), a.device, crits,
                                           a.bins, entropy_batches=ent_b, seed=state.get('seed') or 0)
        rows, summary = compare_drop_sets(model, scores, stats, masks, fraction, a.min_density, positions)
        summary.update(entropy_batches=ent_b, grad_batches=grad_b)
        _print(name, rows, summary, crits)
        # per-channel statistics, for the entropy-vs-sparsity figure (aggregate.py)
        channels = {n: {k: [round(float(v), 5) for v in stats[k][n].tolist()] for k in stats if n in stats[k]}
                    for n in stats.get('H', {})}
        out = os.path.join(a.out_dir, f'overlap_{name}.json')
        with open(out, 'w') as f:
            json.dump({'checkpoint': ck, **meta, 'epoch': state.get('epoch') or meta.get('epoch'),
                       'args': vars(a), 'summary': summary, 'layers': rows, 'channels': channels}, f)
        print(f'wrote {out}\n')
        del model

    if a.compare:
        names = [a.checkpoints[0]] + a.compare
        dens = {}
        for ck in names:
            model, masks, _ = load_checkpoint(ck, 'cpu')
            dens[ck] = {n: float((masks[n] if masks is not None else l.weight != 0).float().mean())
                        for n, l in prunable_layers(model).items()}
        print('per-layer density')
        print(f'{"layer":24s} ' + ' '.join(f'{i:>7d}' for i in range(len(names))))
        for n in dens[names[0]]:
            print(f'{n:24s} ' + ' '.join(f'{dens[k].get(n, float("nan")):7.3f}' for k in names))
        for i, k in enumerate(names):
            print(f'  {i}: {k}')


if __name__ == '__main__':
    main()
