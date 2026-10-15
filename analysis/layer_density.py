"""Per-layer density of the final masks, per criterion and regime (for the allocation figure).

    python analysis/layer_density.py                    # all regimes of the study with final checkpoints

For every regime of experiments/plan.py and every criterion run there, loads the final checkpoints
(results/checkpoints/<run_id>.pt, all seeds found), measures the density of each conv/linear layer
(non-zero weights / weights) and the number of output positions of the layer (its FLOPs per weight),
and writes results/analysis/layer_density.json. Runs on CPU; a few minutes.
"""
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from degst.data import IMAGE_SIZE  # noqa: E402
from degst.sparse import prunable_layers  # noqa: E402
from experiments import plan  # noqa: E402


@torch.no_grad()
def positions(model, hw):
    """Output positions (H*W for conv, 1 for linear) of every conv/linear layer."""
    out, hooks = {}, []
    for n, m in prunable_layers(model).items():
        hooks.append(m.register_forward_hook(
            lambda _m, _i, o, n=n: out.__setitem__(n, o.shape[2] * o.shape[3] if o.dim() == 4 else 1)))
    model.eval()(torch.zeros(1, 3, hw, hw))
    for h in hooks:
        h.remove()
    return out


def main():
    ck_dir = os.path.join(plan.RESULTS, 'checkpoints')
    result, bad = {}, []
    for regime in plan.ALL_REGIMES:
        d, a, sp = regime
        key = f'{d}_{a}_s{round(sp * 100)}'
        methods = {}
        pos, names, numel = None, None, None
        for m in plan.methods_for(regime):
            dens = []
            for s in plan.SEEDS + plan.EXTRA_SEEDS:
                path = os.path.join(ck_dir, plan.regime_cfg(m, regime, s).run_id() + '.pt')
                if not os.path.exists(path):
                    continue
                try:
                    model = torch.load(path, map_location='cpu', weights_only=False)
                except Exception as e:                 # truncated file, e.g. a run killed while saving
                    print(f'  skip unreadable checkpoint {path} ({os.path.getsize(path) / 1e6:.1f} MB): '
                          f'{type(e).__name__}')
                    bad.append(path)
                    continue
                layers = prunable_layers(model)
                if pos is None:
                    pos, names = positions(model, IMAGE_SIZE[d]), list(layers)
                    numel = [l.weight.numel() for l in layers.values()]
                dens.append([float((l.weight != 0).float().mean()) for l in layers.values()])
            if not dens:
                continue
            cols = list(zip(*dens))
            methods[m] = {'n': len(dens), 'density': [statistics.fmean(c) for c in cols],
                          'density_std': [statistics.stdev(c) if len(c) > 1 else 0.0 for c in cols]}
            print(f'{key:24s} {m:18s} seeds={len(dens)}')
        if methods:
            result[key] = {'layers': names, 'positions': [pos[n] for n in names], 'numel': numel,
                           'methods': methods}
    os.makedirs(os.path.join(plan.RESULTS, 'analysis'), exist_ok=True)
    out = os.path.join(plan.RESULTS, 'analysis', 'layer_density.json')
    with open(out, 'w') as f:
        json.dump({**result, '_unreadable_checkpoints': bad}, f, indent=1)
    print(f'wrote {out}')
    if bad:
        print(f'{len(bad)} unreadable checkpoint(s) skipped (their seeds are left out of the means):')
        for p in bad:
            print('   ', p)


if __name__ == '__main__':
    main()
