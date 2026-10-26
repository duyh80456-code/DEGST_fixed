"""Experiment suites of the revision plan, and a resumable runner.

    python -m experiments.plan list c0                 # print the train.py commands
    python -m experiments.plan run c0                  # run them (skips finished runs)
    python -m experiments.plan run pareto --shard 0/2 --device cuda:0   # 1 of 2 GPUs
    python -m experiments.plan status all              # progress of every suite

C0 came out negative (C - A = -0.04 pp, B - A = -2.45 pp), so the paper is now a
study of drop criteria in DST (route 1); the design and its decision rules are
in experiments/DESIGN.md. Suites of that study:
    regimes      3 seeds per cell. Main regimes (ResNet-18 CIFAR-10 90%, CIFAR-100 90%): |w|, H,
                 H|w|, random, Taylor, Fisher, |g|, H_robust|w| (exploratory). Secondary regimes
                 (ResNet-18 CIFAR-10 98%, VGG-16 CIFAR-100 90%): |w|, H, H|w|, random.
                 The |w| seed-0 run of each regime saves mid-training checkpoints.   (72 runs, 9 from c0)
    power        seeds 3 and 4 for |w| and H|w| in every regime (equivalence test)           (16 runs)
    regimes_tin  |w|, H, random on Tiny ImageNet 90%, 3 seeds, + dense (1 seed), seed-major; 100 epochs
                 (= the CIFAR loop in iterations, see TIN_RECIPE), ~1.4 h per run            (10 runs)
    c0_uniform   A/B/C with layer-uniform allocation: separates within-layer ranking from
                 the budget shift between layers, CIFAR-10 90%, 3 seeds                          (9 runs)
    midckpt      the CIFAR-10 90% |w| seed-0 run again (C0 had no mid-training checkpoints),
                 written to results/midckpt                                                      (1 run)
    dense_study  dense references of the regimes, 1 seed                                       (3 runs)
    study        midckpt, c0_uniform, regimes, power, dense_study: the whole study on one GPU
                 with one command, seed-major (92 new runs, ~34 h on an RTX 3090;
                 the seed-0 pass, 28 runs, takes ~10 h)

Original plan (sized for a single consumer GPU, about 190 runs, ~38k epochs in total):
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
import time

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

# --- criteria study (route 1), see experiments/DESIGN.md
# Sized for one GPU: the full set of criteria in two main regimes (easy / hard dataset), the key set
# (|w|, H, H|w|, random) in the two secondary regimes (higher sparsity, second architecture).
KEY_METHODS = ('degst_magnitude', 'degst_entropy', 'degst', 'dst_random')
FULL_METHODS = KEY_METHODS + ('dst_taylor', 'dst_fisher', 'dst_gradient', 'degst_robust')
EXPLORATORY = ('degst_robust',)                # reported, but outside the confirmatory test family
CORE_METHODS = FULL_METHODS
MAIN_REGIMES = (('cifar10', 'resnet18', 0.9), ('cifar100', 'resnet18', 0.9))
SECONDARY_REGIMES = (('cifar10', 'resnet18', 0.98), ('cifar100', 'vgg16', 0.9))
REGIMES = MAIN_REGIMES + SECONDARY_REGIMES
TIN_REGIME = ('tinyimagenet', 'resnet18', 0.9)
# reduced Tiny ImageNet regime (decided 2026-09-30, before any Tiny ImageNet result): tests H0/H1 and
# entropy vs random at a larger scale; no H|w| and no extra seeds, so no equivalence test there
TIN_METHODS = ('degst_magnitude', 'degst_entropy', 'dst_random')
# Tiny ImageNet has 100k training images (781 iterations/epoch vs 390 on CIFAR). Halving every epoch
# count keeps the loop identical in iterations: ~78k SGD steps, warm-up ~3.9k, T_end ~58.6k, ~136
# topology updates at Delta T = 400, and the mid-training checkpoints at the same steps (25/50 = 50/100).
TIN_RECIPE = dict(epochs=100, warmup_epochs=5, end_epoch=75)
TIN_MIDCKPT_EPOCHS = (25, 50)
ALL_REGIMES = REGIMES + (TIN_REGIME,)
EQUIV_METHODS = ('degst_magnitude', 'degst')   # the pair that carries the equivalence claim ...
EXTRA_SEEDS = (3, 4)                           # ... gets 5 seeds in every regime
MIDCKPT_EPOCHS = (50, 100)
MIDCKPT_DIR = os.path.join(RESULTS, 'midckpt')
DENSE_SEEDS = ONE_SEED                         # dense runs are a reference, not a tested arm

# --- phase 2 (experiments/DESIGN.md, "Phase 2"): can activation information beat |w| when used through
# energy rather than entropy? Selection happens on a validation split held out from the training set,
# so the test set is not used to choose a score or beta; only `confirm` reports test accuracy.
SCREEN_REGIMES = (('cifar100', 'resnet18', 0.9), TIN_REGIME)      # largest Dense - |w| gaps (1.95, 1.82 pp)
SCREEN_METHODS = ('degst_magnitude', 'dst_energy', 'dst_fisher', 'dst_composite')   # |w|, S1, S2, S3 (beta 0.5)
VAL_SIZE = {'cifar100': 5000, 'tinyimagenet': 10000}                 # 10% of the training set
BETAS = (0.1, 0.25, 0.5, 1.0, 2.0)                                  # 0.5 is the default of dst_composite
# filled in after the screen, from the pre-specified decision rule: (method, overrides) pairs, run on the
# test set in every regime of ALL_REGIMES with SEEDS
CONFIRM = ()


def screen_cfg(method, regime, seed, **kw):
    """A phase-2 run on the validation split (same loop and recipe as the study, no checkpoints)."""
    d, a, sp = regime
    rec = dict(TIN_RECIPE) if regime == TIN_REGIME else {}
    return cfg(method, sp, seed, dataset=d, model=a, val_size=VAL_SIZE[d], **rec, **kw)


# everything of the criteria study except Tiny ImageNet, for one GPU (run seed-major, see suite())
STUDY = ('midckpt', 'c0_uniform', 'regimes', 'power', 'dense_study')


def methods_for(regime):
    if regime == TIN_REGIME:
        return TIN_METHODS
    return FULL_METHODS if regime in MAIN_REGIMES else KEY_METHODS


def regime_cfg(method, regime, seed):
    """One cell of the study. The |w| seed-0 run of a regime also saves mid-training checkpoints
    (analysis/score_overlap.py); checkpoint_epochs does not change the run id."""
    d, a, sp = regime
    tin = regime == TIN_REGIME
    kw = dict(TIN_RECIPE) if tin else {}
    if method == 'degst_magnitude' and seed == 0:
        kw['checkpoint_epochs'] = TIN_MIDCKPT_EPOCHS if tin else MIDCKPT_EPOCHS
    return cfg(method, sp, seed, dataset=d, model=a, **kw)


def regime_cfg_kw(method, regime, seed, overrides):
    """regime_cfg with extra config overrides (e.g. a composite beta chosen in phase 2)."""
    d, a, sp = regime
    kw = dict(TIN_RECIPE) if regime == TIN_REGIME else {}
    kw.update(overrides)
    return cfg(method, sp, seed, dataset=d, model=a, **kw)


def dense_cfg(dataset, model, seed):
    """Dense reference; trained for the same number of iterations as the sparse runs of its regime."""
    kw = {'epochs': TIN_RECIPE['epochs']} if dataset == TIN_REGIME[0] else {}
    return cfg('dense', 0.0, seed, dataset=dataset, model=model, **kw)

SUITES = ('c0', 'dense', 'study', 'regimes', 'power', 'c0_uniform', 'midckpt', 'dense_study', 'regimes_tin',
          'screen', 'beta_sweep', 'confirm',
          'main', 'pareto', 'sensitivity', 'ext', 'criteria', 'legacy', 'quick')


def cfg(method, sparsity, seed, dataset='cifar10', model='resnet18', **kw):
    out_dir = kw.pop('out_dir', RESULTS)
    return make_config(method, sparsity=sparsity, seed=seed, dataset=dataset, model=model,
                       out_dir=out_dir, **kw)


def midckpt_cfg(dataset, model, sparsity):
    return cfg('degst_magnitude', sparsity, ONE_SEED[0], dataset=dataset, model=model,
               out_dir=MIDCKPT_DIR, checkpoint_epochs=MIDCKPT_EPOCHS)


def pareto_seeds(sparsity):
    return SEEDS if sparsity in MAIN_SPARSITIES else ONE_SEED


def suite(name):
    if name == 'c0':
        return [cfg(m, 0.9, s) for m in C0_METHODS for s in SEEDS]
    if name == 'study':
        out, seen = [], set()
        for s in STUDY:
            for c in suite(s):
                if (c.out_dir, c.run_id()) not in seen:
                    seen.add((c.out_dir, c.run_id()))
                    out.append(c)
        # seed-major order: one full pass over every cell (seed 0) before seeds 1 and 2, so an early
        # look can catch broken cells; the set of runs does not depend on the order
        return sorted(out, key=lambda c: c.seed)
    if name == 'dense_study':
        return [dense_cfg(d, a, s) for d, a in dict.fromkeys((d, a) for d, a, _ in REGIMES) for s in DENSE_SEEDS]
    if name == 'screen':
        jobs = [screen_cfg(m, r, s) for r in SCREEN_REGIMES for m in SCREEN_METHODS for s in SEEDS]
        return sorted(jobs, key=lambda c: c.seed)
    if name == 'beta_sweep':
        jobs = [screen_cfg('dst_composite', r, s, composite_beta=b)
                for r in SCREEN_REGIMES for b in BETAS for s in SEEDS]
        return sorted(jobs, key=lambda c: c.seed)
    if name == 'confirm':
        jobs = [regime_cfg_kw(m, r, s, kw) for m, kw in CONFIRM for r in ALL_REGIMES for s in SEEDS]
        return sorted(jobs, key=lambda c: c.seed)
    if name == 'regimes':
        return [regime_cfg(m, r, s) for r in REGIMES for m in methods_for(r) for s in SEEDS]
    if name == 'power':
        return [regime_cfg(m, r, s) for r in REGIMES for m in EQUIV_METHODS for s in EXTRA_SEEDS]
    if name == 'regimes_tin':
        d, a, _ = TIN_REGIME
        jobs = ([dense_cfg(d, a, s) for s in DENSE_SEEDS] +
                [regime_cfg(m, TIN_REGIME, s) for m in TIN_METHODS for s in SEEDS])
        return sorted(jobs, key=lambda c: c.seed)      # seed-major: complete seed passes first
    if name == 'c0_uniform':
        return [cfg(m, 0.9, s, allocation='uniform') for m in C0_METHODS for s in SEEDS]
    if name == 'midckpt':
        # only regimes whose |w| seed-0 run finished before mid-training checkpoints existed (C0)
        return [midckpt_cfg(*MAIN_REGIMES[0])]
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


def command(c, device=None, num_workers=None):
    args = [sys.executable, 'train.py', '--method', c.method, '--skip-existing']
    if num_workers is not None:
        args += ['--num-workers', str(num_workers)]
    d = c.to_dict()
    keys = ['dataset', 'model', 'sparsity', 'seed', 'out_dir', 'val_size'] + sorted(c.overrides)
    for k in keys:
        if k == 'val_size' and not d[k]:
            continue
        args += ['--' + k.replace('_', '-'), str(d[k])]
    if c.checkpoint_epochs:
        args += ['--checkpoint-epochs', ','.join(str(e) for e in c.checkpoint_epochs)]
    if device:
        args += ['--device', device]
    return args


def done(c):
    return os.path.exists(os.path.join(c.out_dir, c.run_id() + '.json'))


def run_parallel(jobs, n, device=None, num_workers=None):
    """Run up to ``n`` jobs at the same time on one device; each job logs to results/logs/<run_id>.log.

    Every run keeps its own config (batch size, lr, ...), so results are those of a sequential run;
    only wall-clock numbers (epoch time, the speed protocol) are affected by the sharing.
    """
    todo = [c for c in jobs if not done(c)]
    for d in dict.fromkeys(c.dataset for c in todo):    # download once, not from n processes at once
        root = next(c.data_root for c in todo if c.dataset == d)
        subprocess.run([sys.executable, '-c', f'from degst.data import prepare; prepare({d!r}, {root!r})'],
                       check=True)
    log_dir = os.path.join(RESULTS, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    running, failed = [], []

    def wait_until_fewer_than(limit):
        while len(running) >= limit:
            for item in list(running):
                c, proc, fh, t0 = item
                if proc.poll() is None:
                    continue
                running.remove(item)
                fh.close()
                ok = proc.returncode == 0 and done(c)
                if not ok:
                    failed.append(c.run_id())
                print(f'{"done" if ok else "FAIL"}  {c.run_id()}  ({(time.time() - t0) / 60:.0f} min)'
                      f'{"" if ok else "  see " + fh.name}', flush=True)
            if len(running) >= limit:
                time.sleep(10)

    for k, c in enumerate(todo):
        wait_until_fewer_than(n)
        suffix = '' if c.out_dir == RESULTS else '_' + os.path.basename(os.path.normpath(c.out_dir))
        log = open(os.path.join(log_dir, c.run_id() + suffix + '.log'), 'w')
        print(f'start {c.run_id()}{suffix}  [{k + 1}/{len(todo)}]', flush=True)
        running.append((c, subprocess.Popen(command(c, device, num_workers), stdout=log,
                                            stderr=subprocess.STDOUT), log, time.time()))
    wait_until_fewer_than(1)
    print(f'finished {len(todo) - len(failed)}/{len(todo)}' + (f', failed: {failed}' if failed else ''))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['list', 'run', 'status'])
    p.add_argument('suite', help=f'one of {SUITES}, or "all" for status')
    p.add_argument('--shard', default='0/1', help='i/n: run every n-th job starting at i')
    p.add_argument('--device', default=None)
    p.add_argument('--parallel', type=int, default=1,
                   help='runs at the same time on the device (uses spare VRAM; each run is unchanged)')
    p.add_argument('--num-workers', type=int, default=None,
                   help='data-loader workers per run (default 8); lower it with --parallel on few CPU cores')
    a = p.parse_args()
    if a.action == 'status' and a.suite == 'all':
        seen, total_epochs = set(), 0
        for s in SUITES:
            jobs = suite(s)
            new = [c for c in jobs if (c.out_dir, c.run_id()) not in seen]
            seen.update((c.out_dir, c.run_id()) for c in jobs)
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
    if a.action == 'run' and a.parallel > 1:
        run_parallel(jobs, a.parallel, a.device, a.num_workers)
        return
    for c in jobs:
        cmd = command(c, a.device, a.num_workers)
        if a.action == 'list':
            print(('# done  ' if done(c) else '') + ' '.join(cmd[1:]))
        elif not done(c):
            print('>>', ' '.join(cmd[1:]), flush=True)
            subprocess.run(cmd, check=False)


if __name__ == '__main__':
    main()
