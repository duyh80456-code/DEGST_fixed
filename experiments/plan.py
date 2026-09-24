"""Experiment suites of the revision plan, and a resumable runner.

    python -m experiments.plan list c0                 # print the train.py commands
    python -m experiments.plan run c0                  # run them (skips finished runs)
    python -m experiments.plan run pareto --shard 0/2 --device cuda:0   # 1 of 2 GPUs
    python -m experiments.plan status all              # progress of every suite

The plan is sized for a single consumer GPU (about 190 runs, ~38k epochs in total):
    c0           C0.1  entropy ablation: |w| vs H(A) vs H(A)|w|, 90%, 3 seeds     (run first, 9 runs)
    dense              dense references, 3 seeds per dataset/backbone                         (9 runs)
    main         C2.1/C2.4  all methods (incl. RigL) at 50/75/90%, 3 seeds                   (90 runs)
    pareto       C2.2  main + 60/80/95% with 1 seed                                     (+30 runs)
    cost         C2.5  read from the main runs; no extra jobs
    sensitivity  C2.6  bins / update interval / prune ratio / reduction, one factor at a time, 1 seed (7 runs)
    ext          C2.7  CIFAR-100 ResNet-18 and CIFAR-10 MobileNetV2 at 90%, 5 methods, 3 seeds (30 runs)
    criteria     only if C0 is negative ("route 1"): more criteria in the same loop, 90%, 3 seeds
    legacy       optional: pre-revision pipelines at 50/90%, 1 seed, to compare with the old numbers
    quick              5-epoch sanity check on real CIFAR-10

Learning rates are not tuned per method: every method uses the same recipe
(lr 0.1, cosine), i.e. the same, zero, tuning budget. The only exception is the
reproduction of the pre-revision code (degst_v0, lr 0.01 as in that code).

A run is identified by its config (``RunConfig.run_id``), so suites that share a
configuration share the result file and nothing is trained twice.
"""
import argparse
import os
import subprocess
import sys

from degst.config import make_config

RESULTS = 'results'
SEEDS = (0, 1, 2)            # full statistics
ONE_SEED = (0,)              # single-seed points (marked with * in the tables)

C0_METHODS = ('degst_magnitude', 'degst_entropy', 'degst')      # configs A, B, C
MAIN_METHODS = ('degst', 'rigl', 'gmp', 'snip', 'movement',
                'degst_s', 'slimming', 'fpgm', 'autopruner', 'hybrid')
MAIN_SPARSITIES = (0.5, 0.75, 0.9)                 # 3 seeds
EXTRA_PARETO_SPARSITIES = (0.6, 0.8, 0.95)         # 1 seed
PARETO_SPARSITIES = tuple(sorted(MAIN_SPARSITIES + EXTRA_PARETO_SPARSITIES))

SENSITIVITY = [dict(bins=10), dict(bins=100), dict(update_interval=100), dict(update_interval=1000),
               dict(prune_ratio=0.1), dict(prune_ratio=0.3), dict(entropy_reduction='median')]

EXT_SETTINGS = (('cifar100', 'resnet18'), ('cifar10', 'mobilenetv2'))
EXT_METHODS = ('degst', 'rigl', 'gmp', 'degst_s', 'slimming')
EXT_SPARSITIES = (0.9,)

CRITERIA_METHODS = ('degst_magnitude', 'degst_entropy', 'degst', 'dst_taylor', 'dst_fisher',
                    'dst_gradient', 'dst_eltp', 'rigl', 'movement')
CRITERIA_SPARSITIES = (0.9,)

LEGACY_METHODS = ('degst_v0', 'posttrain_magnitude', 'posttrain_layerwise', 'posttrain_snip', 'posttrain_eltp')
LEGACY_SPARSITIES = (0.5, 0.9)

SUITES = ('c0', 'dense', 'main', 'pareto', 'sensitivity', 'ext', 'criteria', 'legacy', 'quick')


def cfg(method, sparsity, seed, dataset='cifar10', model='resnet18', **kw):
    return make_config(method, sparsity=sparsity, seed=seed, dataset=dataset, model=model,
                       out_dir=RESULTS, **kw)


def pareto_seeds(sparsity):
    return SEEDS if sparsity in MAIN_SPARSITIES else ONE_SEED


def suite(name):
    if name == 'c0':
        return [cfg(m, 0.9, s) for m in C0_METHODS for s in SEEDS]
    if name == 'dense':
        return [cfg('dense', 0.0, s, dataset=d, model=a)
                for d, a in (('cifar10', 'resnet18'),) + EXT_SETTINGS for s in SEEDS]
    if name == 'main':
        return [cfg(m, sp, s) for m in MAIN_METHODS for sp in MAIN_SPARSITIES for s in SEEDS]
    if name == 'pareto':
        return [cfg(m, sp, s) for m in MAIN_METHODS for sp in PARETO_SPARSITIES for s in pareto_seeds(sp)]
    if name == 'sensitivity':
        return [cfg('degst', 0.9, s, **o) for o in SENSITIVITY for s in ONE_SEED]
    if name == 'ext':
        return [cfg(m, sp, s, dataset=d, model=a) for d, a in EXT_SETTINGS
                for m in EXT_METHODS for sp in EXT_SPARSITIES for s in SEEDS]
    if name == 'criteria':
        return [cfg(m, sp, s) for m in CRITERIA_METHODS for sp in CRITERIA_SPARSITIES for s in SEEDS]
    if name == 'legacy':
        return [cfg(m, sp, s) for m in LEGACY_METHODS for sp in LEGACY_SPARSITIES for s in ONE_SEED]
    if name == 'quick':
        return [make_config(m, sparsity=0.0 if m == 'dense' else 0.9, seed=0, epochs=5, warmup_epochs=1,
                            end_epoch=4, prune_epoch=3, out_dir=os.path.join(RESULTS, 'quick'))
                for m in ('dense', 'degst', 'rigl', 'degst_s')]
    if name == 'cost':
        return []
    raise KeyError(f'unknown suite {name}; choose from {SUITES}')


def command(c, device=None):
    args = [sys.executable, 'train.py', '--method', c.method, '--skip-existing']
    d = c.to_dict()
    keys = ['dataset', 'model', 'sparsity', 'seed', 'out_dir', 'val_size'] + sorted(c.overrides)
    for k in keys:
        if k == 'val_size' and not d[k]:
            continue
        args += ['--' + k.replace('_', '-'), str(d[k])]
    if device:
        args += ['--device', device]
    return args


def done(c):
    return os.path.exists(os.path.join(c.out_dir, c.run_id() + '.json'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['list', 'run', 'status'])
    p.add_argument('suite', help=f'one of {SUITES}, or "all" for status')
    p.add_argument('--shard', default='0/1', help='i/n: run every n-th job starting at i')
    p.add_argument('--device', default=None)
    a = p.parse_args()
    if a.action == 'status' and a.suite == 'all':
        seen, total_epochs = set(), 0
        for s in SUITES:
            jobs = suite(s)
            new = [c for c in jobs if c.run_id() not in seen]
            seen.update(c.run_id() for c in jobs)
            total_epochs += sum(c.epochs for c in new if s != 'quick')
            print(f'{s:12s} {sum(done(c) for c in jobs):4d}/{len(jobs):<4d} done  ({len(new)} not shared with earlier suites)')
        print(f'unique runs {len(seen)}, epochs excluding quick {total_epochs}')
        return
    jobs = suite(a.suite)
    i, n = (int(x) for x in a.shard.split('/'))
    jobs = jobs[i::n]
    if a.action == 'status':
        print(f'{a.suite}: {sum(done(c) for c in jobs)}/{len(jobs)} done')
        return
    for c in jobs:
        cmd = command(c, a.device)
        if a.action == 'list':
            print(('# done  ' if done(c) else '') + ' '.join(cmd[1:]))
        elif not done(c):
            print('>>', ' '.join(cmd[1:]), flush=True)
            subprocess.run(cmd, check=False)


if __name__ == '__main__':
    main()
