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
