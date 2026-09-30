"""Re-measure speed of saved checkpoints under the protocol in degst/metrics.py.

    python bench.py results/checkpoints/*.pt --batches 1,128,512,1024 --device cuda

Useful when training and timing happen on different machines: speed numbers in
one table must all come from the same GPU. Writes results/bench_<gpu>.json.
"""
import argparse
import json
import os

import torch

from degst import metrics
from degst.data import IMAGE_SIZE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoints', nargs='+')
    p.add_argument('--batches', default='1,128,1024')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--iters', type=int, default=200)
    a = p.parse_args()
    batches = tuple(int(b) for b in a.batches.split(','))
    proto = metrics.protocol(a.device, a.warmup, a.iters, batches)
    out = {'protocol': proto, 'runs': {}}
    for ck in a.checkpoints:
        model = torch.load(ck, map_location=a.device, weights_only=False)
        rid = os.path.splitext(os.path.basename(ck))[0]
        hw = IMAGE_SIZE.get(rid.split('_')[0], 32)   # run ids start with the dataset name
        out['runs'][rid] = {**metrics.param_counts(model), **metrics.flops(model, (1, 3, hw, hw)),
                            'speed': metrics.measure_speed(model, batches, a.device, a.warmup, a.iters, hw)}
        s = out['runs'][rid]['speed']
        print(rid, {b: round(v.get('latency_ms_median', float('nan')), 3) for b, v in s.items()})
    name = proto['device'].replace(' ', '_')
    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', f'bench_{name}.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
