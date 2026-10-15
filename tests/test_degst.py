"""Tests for the degst package (replaces legacy test.py and demo_eltp_quick.py).

    python tests/test_degst.py            # unit tests, ~30 s on CPU, no dataset needed
    python tests/test_degst.py --smoke    # additionally trains every registered method briefly on random data
    pytest tests/                         # same unit tests under pytest
"""
import math
import os
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from degst.config import METHODS, make_config  # noqa: E402
from degst.criteria import eltp_score  # noqa: E402
from degst.entropy import channel_entropy, expand_to_weight  # noqa: E402
from degst.metrics import flops, param_counts  # noqa: E402
from degst.models import build_model  # noqa: E402
from degst.sparse import MaskManager  # noqa: E402
from degst.structured import ChannelGroups  # noqa: E402


def test_models_forward():
    for name in ('resnet18', 'vgg16', 'mobilenetv2'):
        assert build_model(name, 10)(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    for name in ('resnet18', 'vgg16', 'mobilenetv2'):   # Tiny ImageNet: 64x64, 200 classes
        assert build_model(name, 200)(torch.randn(2, 3, 64, 64)).shape == (2, 200)
    m = build_model('resnet18', 10)
    assert param_counts(m)['weights_total'] == 11_164_352
    assert flops(m)['flops_dense'] == 1_110_845_440


def test_entropy():
    x = torch.randn(8, 4, 5, 5)
    x[:, 1] = 3.0                                    # constant channel -> zero entropy
    h = channel_entropy(x, bins=30)
    assert abs(h[1].item()) < 1e-6 and (h[[0, 2, 3]] > 2).all() and (h <= math.log(30) + 1e-5).all()
    w = torch.randn(6, 4, 3, 3)
    assert torch.equal(expand_to_weight(h, w)[5, 3], torch.full((3, 3), h[3].item()))
    wd = torch.randn(4, 1, 3, 3)                     # depthwise
    assert all(expand_to_weight(h, wd, groups=4)[c, 0, 0, 0] == h[c] for c in range(4))
    wg = torch.randn(6, 2, 3, 3)                     # 2 groups
    eg = expand_to_weight(h, wg, groups=2)
    assert eg[4, 1, 0, 0] == h[3] and eg[0, 1, 0, 0] == h[1]
    assert expand_to_weight(h, torch.randn(10, 4))[7, 2] == h[2]


def test_eltp_matches_legacy_formula():
    """ELTP score = |w g| (1 - E_i / log n), E_i = -p_i log p_i, p = |w| / sum |w| (archive/legacy_code.zip: utils/pruning.py)."""
    torch.manual_seed(0)
    w, g = torch.randn(5, 7), torch.randn(5, 7)
    p = w.abs().flatten() / w.abs().sum()
    e = (-p * torch.log(p + 1e-12)).view_as(w)
    ref = (w * g).abs() * (1 - e / math.log(w.numel()))
    assert torch.allclose(eltp_score(w, g), ref, atol=1e-6)


def test_mask_manager_exact_sparsity():
    torch.manual_seed(0)
    for alloc in ('global', 'uniform', 'erk'):
        m = build_model('resnet18', 10)
        mm = MaskManager(m, alloc)
        scores = {n: l.weight.detach().abs() for n, l in mm.layers.items()}
        mm.init_from_scores(scores, 0.9)
        assert abs(mm.sparsity() - 0.9) < 2e-3, (alloc, mm.sparsity())
        assert all(mask.sum() > 0 for mask in mm.masks.values())
        for l in mm.layers.values():
            l.weight.grad = torch.randn_like(l.weight)
        nnz = mm.nnz()
        info = mm.prune_regrow(scores, {n: l.weight.grad.abs() for n, l in mm.layers.items()}, 0.2)
        assert mm.nnz() == nnz and info['pruned'] == info['grown'] > 0
        assert param_counts(m)['weights_nonzero'] <= nnz     # grown weights start at zero


def test_physical_channel_removal_is_exact():
    torch.manual_seed(0)
    for name in ('resnet18', 'vgg16', 'mobilenetv2'):
        m = build_model(name, 10).eval()
        for mod in m.modules():                      # non-trivial BN statistics
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.normal_()
                mod.running_var.uniform_(0.5, 2)
                mod.weight.data.normal_()
                mod.bias.data.normal_()
        x = torch.randn(4, 3, 32, 32)
        cg = ChannelGroups(m, x[:1], m.linear)
        assert cg.params_for(cg.size) == cg.dense_numel
        target = int(0.3 * cg.dense_numel)
        cg.prune_to_params([torch.rand(c) for c in cg.size], target)
        predicted = cg.params_for(cg.kept())
        cg.refresh_gates()
        with torch.no_grad():
            y_gated = m(x)
            compact = cg.compact()
            y_compact = compact(x)
        assert param_counts(compact)['weights_total'] == predicted <= target
        assert (y_gated - y_compact).abs().max() < 1e-4, name


def test_score_overlap_diagnostics():
    """Constant entropy: H*|w| ranks exactly like |w|, so both drop the same weights."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis'))
    from score_overlap import collect, compare_drop_sets, parse_name
    from degst.criteria import ScoreState
    torch.manual_seed(0)
    m = build_model('resnet18', 10)
    mm = MaskManager(m, 'global')
    mm.init_from_scores({n: l.weight.detach().abs() for n, l in mm.layers.items()}, 0.9)

    def scores(entropy):
        return {c: ScoreState(mm.layers, c).scores(entropy) for c in ('magnitude', 'entropy_magnitude')}

    const = {n: torch.full((l.weight.shape[1],), 2.0) for n, l in mm.layers.items()}
    _, s = compare_drop_sets(m, scores(const), {'H': const}, fraction=0.2)
    assert s['jaccard']['magnitude']['entropy_magnitude'] == 1.0 and s['k_dropped'] > 0
    assert s['jaccard_within_vs_ref']['entropy_magnitude'] == 1.0
    varied = {n: torch.rand(l.weight.shape[1]) + 0.5 for n, l in mm.layers.items()}
    rows, s = compare_drop_sets(m, scores(varied), {'H': varied}, fraction=0.2)
    assert s['jaccard']['magnitude']['entropy_magnitude'] < 1.0
    assert sum(r['drop']['entropy_magnitude'] for r in rows) == sum(r['drop']['magnitude'] for r in rows) == s['k_dropped']

    # full path: 3 batches, entropy from the first only, gradient statistics from all, explicit masks
    data = [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))) for _ in range(3)]
    sc, stats, pos = collect(m, data, 'cpu', entropy_batches=1)
    assert set(sc) == {'magnitude', 'entropy', 'entropy_magnitude', 'entropy_magnitude_robust', 'taylor',
                       'fisher', 'gradient', 'random'}
    assert set(stats) == {'H', 'H_robust', 'zero'}
    rows, s = compare_drop_sets(m, sc, stats, masks=mm.masks, fraction=0.2, positions=pos)
    assert all(abs(s['jaccard'][c][c] - 1.0) < 1e-9 for c in sc) and 0 < s['flops_dropped_pct']['magnitude'] < 100
    assert 'spearman_H_zero_pooled' in s and 0 <= rows[1]['zero_share'] <= 1
    assert parse_name('cifar100_resnet18_degst_magnitude_s98_seed0_ep50') == dict(
        dataset='cifar100', model='resnet18', method='degst_magnitude', sparsity_pct=98, seed=0, epoch=50)


def test_random_criterion_and_robust_entropy():
    from degst.criteria import ScoreState
    from degst.entropy import channel_entropy, channel_entropy_robust
    m = build_model('resnet18', 10)
    layers = MaskManager(m).layers
    a, b, c = (ScoreState(layers, 'random', seed=s).scores() for s in (0, 0, 1))
    n = next(iter(a))
    assert torch.equal(a[n], b[n]) and not torch.equal(a[n], c[n])
    state = torch.random.get_rng_state()
    ScoreState(layers, 'random', seed=0).scores()
    assert torch.equal(state, torch.random.get_rng_state())          # global RNG (data order) untouched
    mm = MaskManager(m, 'global')
    mm.init_from_scores(a, 0.9)
    assert abs(mm.sparsity() - 0.9) < 2e-3

    torch.manual_seed(0)
    x = torch.rand(8, 3, 16, 16)
    x[:, 1] = torch.where(torch.rand(8, 16, 16) < 0.7, torch.zeros(8, 16, 16), x[:, 1])   # 70 % ReLU zeros
    x[0, 2, 0, 0] = 1000.0                                                                  # one outlier
    x[:, 0] = 0.0                                                                           # dead channel
    h, hr = channel_entropy(x, 30), channel_entropy_robust(x, 30)
    assert abs(hr[0].item()) < 1e-6
    assert hr[1] > h[1] + 0.5 and hr[2] > h[2] + 2 and hr[2] > math.log(30) - 0.3   # zeros / outlier removed


def test_statistics():
    import aggregate
    ref = [95.10, 95.20, 95.15, 95.12, 95.18]
    same = aggregate.compare_to(ref, [95.14, 95.11, 95.19, 95.16, 95.13], margin=0.3)
    assert same['equivalent'] and same['p'] > 0.5 and same['ci95'][0] < 0 < same['ci95'][1]
    worse = aggregate.compare_to(ref, [93.1, 93.2, 93.0, 93.15, 93.05], margin=0.3)
    assert not worse['equivalent'] and worse['p'] < 1e-4 and worse['ci95'][1] < -1.5
    assert same['n'] == 5 and aggregate.compare_to(ref[:1], ref) is None
    adj = aggregate.holm({'a': 0.01, 'b': 0.04, 'c': 0.03})
    assert abs(adj['a'] - 0.03) < 1e-12 and abs(adj['c'] - 0.06) < 1e-12 and abs(adj['b'] - 0.06) < 1e-12


def test_plan_suites_share_runs():
    """New suites reuse finished C0 runs, and analysis-only options never change a run id."""
    from experiments import plan
    c0 = {c.run_id() for c in plan.suite('c0')}
    regimes = plan.suite('regimes')
    assert len(regimes) == 2 * 8 * 3 + 2 * 4 * 3 and c0 <= {c.run_id() for c in regimes}
    assert len(plan.suite('power')) == 16 and len(plan.suite('dense_study')) == 3
    # mid-training checkpoints: exactly the |w| seed-0 run of every regime (+ the C0 one again in midckpt)
    ck = [c for c in regimes if c.checkpoint_epochs]
    assert len(ck) == len(plan.REGIMES) and all(c.method == 'degst_magnitude' and c.seed == 0 for c in ck)
    study = plan.suite('study')
    keys = [(c.out_dir, c.run_id()) for c in study]
    assert len(keys) == len(set(keys)) == 101 and sum(d != 'results' or r not in c0 for d, r in keys) == 92
    assert [c.seed for c in study] == sorted(c.seed for c in study)   # seed 0 of every cell first
    assert not c0 & {c.run_id() for c in plan.suite('c0_uniform')}
    # Tiny ImageNet: 10 runs, seed-major, 100 epochs with the loop matched to CIFAR in iterations
    tin = plan.suite('regimes_tin')
    assert len(tin) == 10 and [c.seed for c in tin] == sorted(c.seed for c in tin)
    assert all(c.epochs == 100 for c in tin)
    assert all((c.warmup_epochs, c.end_epoch) == (5, 75) for c in tin if c.method != 'dense')
    ck = [c for c in tin if c.checkpoint_epochs]
    assert len(ck) == 1 and ck[0].checkpoint_epochs == (25, 50)
    # the tables look runs up with the same configs as the runner (same run ids)
    assert {c.run_id() for c in tin} == ({plan.regime_cfg(m, plan.TIN_REGIME, s).run_id()
                                          for m in plan.TIN_METHODS for s in plan.SEEDS}
                                         | {plan.dense_cfg('tinyimagenet', 'resnet18', 0).run_id()})
    # CIFAR run ids unchanged by the Tiny ImageNet recipe
    assert plan.dense_cfg('cifar10', 'resnet18', 0).run_id() == 'cifar10_resnet18_dense_s0_seed0'
    assert plan.regime_cfg('degst_magnitude', plan.TIN_REGIME, 0).run_id().endswith('_c8231824')


def test_runner_and_train_agree_on_run_ids():
    """The run id the plan expects (done(), aggregate.py) equals the one train.py writes from the CLI."""
    import train
    from experiments import plan
    for name in ('c0', 'study', 'regimes_tin', 'criteria'):
        for c in plan.suite(name):
            args = vars(train._parser().parse_args(plan.command(c)[2:]))
            method = args.pop('method')
            args.pop('device'), args.pop('skip_existing')
            assert make_config(method, **args).run_id() == c.run_id(), (name, c.run_id())
    a = make_config('degst_magnitude', sparsity=0.9, seed=0)
    b = make_config('degst_magnitude', sparsity=0.9, seed=0, checkpoint_epochs=(50, 100), eval_every=1)
    assert a.run_id() == b.run_id() and '--checkpoint-epochs' in plan.command(plan.suite('midckpt')[0])


def test_mid_training_checkpoint():
    from degst.trainer import run
    out = tempfile.mkdtemp(prefix='degst_ckpt_')
    cfg = make_config('degst_magnitude', dataset='fake', sparsity=0.9, epochs=3, warmup_epochs=0, end_epoch=2.5,
                      update_interval=2, batch_size=64, checkpoint_epochs=(1,), out_dir=out, eval_every=5,
                      speed_batches=(1,), speed_warmup=1, speed_iters=2)
    r = run(cfg, 'cpu', verbose=False)
    ck = torch.load(os.path.join(out, 'checkpoints', cfg.run_id() + '_ep1.pt'), weights_only=False)
    assert ck['epoch'] == 1 and isinstance(ck['model'], nn.Module)
    assert 0 < ck['swap_fraction'] <= 0.2 and ck['update_interval'] == 2 and ck['entropy_batches'] == 4
    density = sum(v.sum().item() for v in ck['masks'].values()) / sum(v.numel() for v in ck['masks'].values())
    assert abs(density - 0.1) < 2e-3
    accs = [h['test_acc'] for h in r['history']]           # epochs 1 and 3 (last) evaluated, 2 skipped
    assert accs[0] is not None and accs[1] is None and accs[2] is not None and r['final_acc'] == accs[2]


def test_tinyimagenet_reader():
    """Folder layout of tiny-imagenet-200: train/<wnid>/images/, val/images/ + val_annotations.txt."""
    from PIL import Image
    from degst.data import TinyImageNet
    root = tempfile.mkdtemp(prefix='degst_tin_')
    d = os.path.join(root, 'tiny-imagenet-200')
    wnids = ['n02', 'n01']
    for w in wnids:
        os.makedirs(os.path.join(d, 'train', w, 'images'))
        for i in range(3):
            Image.new('L', (64, 64)).save(os.path.join(d, 'train', w, 'images', f'{w}_{i}.JPEG'))
    os.makedirs(os.path.join(d, 'val', 'images'))
    rows = []
    for i, w in enumerate(wnids * 2):
        Image.new('RGB', (64, 64)).save(os.path.join(d, 'val', 'images', f'val_{i}.JPEG'))
        rows.append(f'val_{i}.JPEG\t{w}\t0\t0\t63\t63\n')
    with open(os.path.join(d, 'wnids.txt'), 'w') as f:
        f.write('\n'.join(wnids) + '\n')
    with open(os.path.join(d, 'val', 'val_annotations.txt'), 'w') as f:
        f.writelines(rows)
    tr, va = TinyImageNet(root, train=True), TinyImageNet(root, train=False)
    assert tr.classes == ['n01', 'n02'] and len(tr) == 6 and sorted(tr.targets) == [0, 0, 0, 1, 1, 1]
    assert va.targets == [1, 0, 1, 0] and tr[0][0].mode == 'RGB'
    # decoded-image cache: built once, same pixels as decoding the JPEG
    from torchvision.datasets.folder import default_loader
    import numpy as np
    assert os.path.isfile(tr.cache) and os.path.isfile(va.cache)
    assert np.array_equal(np.asarray(tr[4][0]), np.asarray(default_loader(tr.samples[4][0])))
    assert tr[4][1] == tr.samples[4][1] and TinyImageNet(root, train=True)._cache_ok()


def smoke_all_methods():
    out = tempfile.mkdtemp(prefix='degst_smoke_')
    from degst.trainer import run
    failed = []
    for method in sorted(METHODS):
        cfg = make_config(method, dataset='fake', sparsity=0.0 if method == 'dense' else 0.9, epochs=3,
                          warmup_epochs=1, end_epoch=2.5, prune_epoch=1, update_interval=2, entropy_batches=2,
                          out_dir=out, speed_batches=(1,), speed_warmup=1, speed_iters=2, save_checkpoint=False)
        try:
            r = run(cfg, 'cpu', verbose=False)
            sp = r['final']['sparsity_actual']
            ok = abs(sp - cfg.sparsity) < 5e-3
            print(f'{"ok  " if ok else "FAIL"} {method:22s} sparsity {sp:.4f}')
            if not ok:
                failed.append(method)
        except Exception as e:  # report every method, then fail
            print(f'FAIL {method:22s} {type(e).__name__}: {e}')
            failed.append(method)
    return failed


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f'ok   {t.__name__}')
    if '--smoke' in sys.argv and smoke_all_methods():
        sys.exit(1)
    print('all passed')
