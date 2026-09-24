"""Run one training job and write results/<run_id>.json.

    python train.py --method degst --sparsity 0.9 --seed 0
    python train.py --method rigl --sparsity 0.9 --seed 0 --lr 0.1
    python train.py --method degst --dataset fake --epochs 2 --end-epoch 1.5 --warmup-epochs 0 \
        --update-interval 2 --out-dir /tmp/smoke          # smoke test on CPU

Any RunConfig field can be overridden with --field-name value; see degst/config.py.
"""
import argparse
import dataclasses
import os
import sys

from degst.config import METHODS, RunConfig, make_config
from degst.trainer import run


def _parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--method', required=True, choices=sorted(METHODS))
    p.add_argument('--device', default=None)
    p.add_argument('--skip-existing', action='store_true', help='do nothing if the result JSON exists')
    for f in dataclasses.fields(RunConfig):
        if f.name in ('method', 'overrides'):
            continue
        flag = '--' + f.name.replace('_', '-')
        if f.type in (bool, 'bool'):
            p.add_argument(flag, type=lambda s: s.lower() in ('1', 'true', 'yes'), default=None)
        elif f.name == 'speed_batches':
            p.add_argument(flag, type=lambda s: tuple(int(x) for x in s.split(',')), default=None)
        else:
            typ = {int: int, float: float, str: str, 'int': int, 'float': float, 'str': str}[f.type]
            p.add_argument(flag, type=typ, default=None)
    return p


def main(argv=None):
    args = vars(_parser().parse_args(argv))
    method, device, skip = args.pop('method'), args.pop('device'), args.pop('skip_existing')
    cfg = make_config(method, **args)
    if skip and os.path.exists(os.path.join(cfg.out_dir, cfg.run_id() + '.json')):
        print(f'skip {cfg.run_id()} (exists)')
        return 0
    run(cfg, device)
    return 0


if __name__ == '__main__':
    sys.exit(main())
