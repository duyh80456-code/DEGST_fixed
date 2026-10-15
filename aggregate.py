"""Aggregate results/*.json into the paper's tables and figures.

    python aggregate.py            # criteria study: c0_ablation, regimes, allocation, entropy_vs_random tables,
                                   # divergence_accuracy figure, results/*_summary.json
    python aggregate.py --all      # + the tables/figures of the pre-C0 plan (main, Pareto, speed, ...)

Every cell is mean +- std over seeds. Missing runs render as "--", so the paper
compiles at any stage; the number of seeds found is printed for every cell
group. Accuracy is always the final-epoch test accuracy.

C0 decision rule (from the revision plan): entropy is considered useful only if
mean(C) - mean(A) > 1.96 * pooled std, pooled std = sqrt((sd_A^2 + sd_C^2) / 2).
A Welch t statistic is printed alongside; with 3 seeds the rule is a coarse
screen, not a significance test.
"""
import json
import math
import os
import statistics

from degst.config import METHODS
from experiments import plan

TABLES = os.path.join('paper', 'tables')
FIGS = os.path.join('paper', 'Figures')

NAMES = {
    'dense': 'Dense', 'degst': 'DEGST', 'degst_entropy': 'DEGST, $H(A)$ only',
    'degst_magnitude': 'DEGST loop, $|w|$ only', 'dst_taylor': 'Taylor $|w g|$', 'dst_fisher': 'Fisher $(wg)^2/2$',
    'dst_gradient': 'Gradient $|g|$', 'rigl': 'RigL', 'gmp': 'GMP', 'snip': 'SNIP', 'movement': 'Movement',
    'degst_s': 'DEGST-S', 'slimming': 'Network Slimming', 'fpgm': 'FPGM', 'l1filter': '$\\ell_1$-filter',
    'autopruner': 'AutoPruner$^\\dagger$', 'hybrid': 'DEGST-Hybrid',
    'dst_eltp': 'ELTP $|wg|(1-E/H_{\\max})$', 'degst_v0': 'DEGST v0 (pre-revision)',
    'posttrain_magnitude': 'Magnitude + LTH rewind', 'posttrain_layerwise': 'Layer-wise magnitude + LTH rewind',
    'posttrain_snip': 'SNIP (trained) + LTH rewind', 'posttrain_eltp': 'ELTP (init) + LTH rewind',
}
KIND = {'degst': 'U', 'rigl': 'U', 'gmp': 'U', 'snip': 'U', 'movement': 'U', 'degst_s': 'S', 'slimming': 'S',
        'fpgm': 'S', 'autopruner': 'S', 'hybrid': 'H'}


def load(c):
    path = os.path.join(c.out_dir, c.run_id() + '.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def collect(cfgs):
    """Group configs that differ only by seed; return {key: [result, ...]}."""
    groups = {}
    for c in cfgs:
        key = c.run_id().replace(f'_seed{c.seed}', '')
        groups.setdefault(key, {'cfg': c, 'runs': []})
        r = load(c)
        if r is not None:
            groups[key]['runs'].append(r)
    return groups


def ms(vals, fmt='{:.2f}', scale=1.0):
    vals = [v * scale for v in vals if v is not None]
    if not vals:
        return '--'
    if len(vals) == 1:
        return fmt.format(vals[0]) + '$^{*}$'
    return fmt.format(statistics.fmean(vals)) + ' $\\pm$ ' + fmt.format(statistics.stdev(vals))


def mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def stdev(vals):
    vals = [v for v in vals if v is not None]
    return statistics.stdev(vals) if len(vals) > 1 else None


def field(runs, path):
    out = []
    for r in runs:
        x = r
        for p in path:
            x = x.get(p) if isinstance(x, dict) else None
        out.append(x)
    return out


def write(name, text):
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, name), 'w', encoding='utf8') as f:
        f.write(text)
    print(f'wrote {os.path.join(TABLES, name)}')


def table(caption, label, colspec, header, rows, note=''):
    body = '\n'.join(' & '.join(r) + ' \\\\' for r in rows)
    note = f'\n\\par\\smallskip\\footnotesize {note}' if note else ''
    return (f'\\begin{{table*}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n'
            f'\\resizebox{{\\textwidth}}{{!}}{{%\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n'
            f'\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}}}{note}\n\\end{{table*}}\n')


SEED_NOTE = ('Mean $\\pm$ std over 3 seeds; $^{*}$ marks single-seed cells (no std); -- = not yet run. '
             'Accuracy is final-epoch test accuracy.')


# --------------------------------------------------------------------------- C0
def c0():
    g = collect(plan.suite('c0'))
    by = {v['cfg'].method: [r['final_acc'] for r in v['runs']] for v in g.values()}
    rows = []
    for m, lab in zip(plan.C0_METHODS, ('A', 'B', 'C')):
        runs = next(v['runs'] for v in g.values() if v['cfg'].method == m)
        rows.append([lab, {'degst_magnitude': '$|w|$', 'degst_entropy': '$H(A)$',
                           'degst': '$H(A)\\cdot|w|$'}[m],
                     ms(by[m]), ms(field(runs, ('final', 'sparsity_actual')), '{:.2f}', 100),
                     ms(field(runs, ('final', 'flops_reduction_sparse')), '{:.1f}', 100),
                     ms(field(runs, ('final', 'weights_nonzero')), '{:,.0f}'),
                     str(len(runs))])
    decision = {'n': {m: len(v) for m, v in by.items()}}
    a, cc = by['degst_magnitude'], by['degst']
    if len(a) >= 2 and len(cc) >= 2:
        pooled = math.sqrt((statistics.variance(a) + statistics.variance(cc)) / 2)
        diff = statistics.fmean(cc) - statistics.fmean(a)
        se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(cc) / len(cc))
        decision.update(mean_A=statistics.fmean(a), mean_C=statistics.fmean(cc), diff_C_minus_A=diff,
                        pooled_std=pooled, threshold=1.96 * pooled, welch_t=diff / se if se > 0 else None,
                        entropy_helps=diff > 1.96 * pooled)
        if len(by['degst_entropy']) >= 2:
            decision['diff_B_minus_A'] = statistics.fmean(by['degst_entropy']) - statistics.fmean(a)
        verdict = ('POSITIVE: entropy helps -> continue with A1-A4' if decision['entropy_helps'] else
                   'NEGATIVE: no benefit over |w| -> criteria study (experiments/DESIGN.md)')
        print(f'C0: C-A = {diff:+.3f} pp, threshold 1.96*pooled = {1.96 * pooled:.3f} pp  => {verdict}')
    else:
        print('C0: not enough runs for a decision', decision['n'])
    os.makedirs(plan.RESULTS, exist_ok=True)
    with open(os.path.join(plan.RESULTS, 'c0_decision.json'), 'w') as f:
        json.dump(decision, f, indent=1)
    write('c0_ablation.tex', table(
        'Ablation C0: the same prune/regrow loop with three drop criteria (CIFAR-10, ResNet-18, 90\\% '
        'target sparsity). Only the importance score differs between rows.',
        'tab:c0', 'llccccc',
        'Cfg & Drop score & Acc (\\%) & Sparsity (\\%) & FLOPs red.\\ sparse (\\%) & Non-zero weights & Seeds',
        rows, SEED_NOTE))
    return decision


# --------------------------------------------------------------------------- main table
def main_table(dataset='cifar10', model='resnet18', sparsities=plan.MAIN_SPARSITIES, fname='main_results.tex',
               label='tab:main'):
    cfgs = [plan.cfg(m, sp, s, dataset=dataset, model=model) for sp in sparsities
            for m in plan.MAIN_METHODS for s in plan.SEEDS]
    g = collect(cfgs)
    dense = collect([plan.cfg('dense', 0.0, s, dataset=dataset, model=model) for s in plan.SEEDS])
    rows = []
    for v in dense.values():
        rows.append(['--', 'Dense', '', ms([r['final_acc'] for r in v['runs']]), '0.00',
                     ms(field(v['runs'], ('final', 'weights_total')), '{:,.0f}'), '0.0', '0.0',
                     ms(field(v['runs'], ('final', 'speed', '1', 'latency_ms_median')), '{:.3f}'),
                     ms(field(v['runs'], ('final', 'speed', '1024', 'throughput_img_s')), '{:,.0f}')])
    for v in g.values():
        c, runs = v['cfg'], v['runs']
        rows.append([f'{round(c.sparsity * 100)}\\%', NAMES[c.method], KIND.get(c.method, ''),
                     ms([r['final_acc'] for r in runs]),
                     ms(field(runs, ('final', 'sparsity_actual')), '{:.2f}', 100),
                     ms(field(runs, ('final', 'weights_total')), '{:,.0f}'),
                     ms(field(runs, ('final', 'flops_reduction_sparse')), '{:.1f}', 100),
                     ms(field(runs, ('final', 'flops_reduction_dense_arch')), '{:.1f}', 100),
                     ms(field(runs, ('final', 'speed', '1', 'latency_ms_median')), '{:.3f}'),
                     ms(field(runs, ('final', 'speed', '1024', 'throughput_img_s')), '{:,.0f}')])
    write(fname, table(
        f'{dataset.upper().replace("CIFAR", "CIFAR-")} / {model}: all methods at the same weight budget. '
        'Weights = conv/linear weights stored in the final architecture (physically removed channels are '
        'not counted). FLOPs red.\\ (sparse) counts only non-zero weights; FLOPs red.\\ (dense) counts the '
        'architecture as executed by dense kernels. Latency at batch 1 and throughput at batch 1024 come '
        'from the protocol of Sec.~\\ref{sec:speed}.',
        label, 'lllccccccc',
        'Target & Method & Type & Acc (\\%) & Sparsity (\\%) & Weights & FLOPs red.\\ sparse (\\%) & '
        'FLOPs red.\\ dense (\\%) & Latency b=1 (ms) & Throughput b=1024 (img/s)',
        rows, SEED_NOTE + ' Type: U = unstructured, S = structured, H = hybrid. '
        '$^\\dagger$AutoPruner-style learned channel gates (input-independent simplification).'))


# --------------------------------------------------------------------------- speed & cost
def speed_table():
    cfgs = [plan.cfg('dense', 0.0, s) for s in plan.SEEDS] + [plan.cfg(m, 0.9, s) for m in plan.MAIN_METHODS for s in plan.SEEDS]
    g = collect(cfgs)
    rows = []
    for v in g.values():
        c, runs = v['cfg'], v['runs']
        row = [NAMES[c.method]]
        for b in ('1', '128', '1024'):
            row += [ms(field(runs, ('final', 'speed', b, 'latency_ms_median')), '{:.3f}'),
                    ms(field(runs, ('final', 'speed', b, 'throughput_img_s')), '{:,.0f}')]
        rows.append(row)
    write('speed.tex', table(
        'Inference latency (median ms per forward pass) and throughput (images/s = batch / median latency) '
        'at 90\\% target sparsity, CIFAR-10 / ResNet-18. Both numbers of a pair come from the same '
        'measurement.', 'tab:speed', 'lcccccc',
        'Method & Lat.\\ b=1 & Thr.\\ b=1 & Lat.\\ b=128 & Thr.\\ b=128 & Lat.\\ b=1024 & Thr.\\ b=1024',
        rows, SEED_NOTE))


def cost_table():
    methods = ('dense', 'degst', 'rigl', 'gmp', 'movement', 'degst_s', 'hybrid')
    cfgs = [plan.cfg(m, 0.0 if m == 'dense' else 0.9, s) for m in methods for s in plan.SEEDS]
    g = collect(cfgs)
    rows = []
    for v in g.values():
        c, runs = v['cfg'], v['runs']
        overhead = [(r['cost']['entropy_hook_seconds'] / r['cost']['train_seconds']) for r in runs]
        upd = [(r['cost']['topology_update_seconds'] / r['cost']['train_seconds']) for r in runs]
        rows.append([NAMES[c.method], ms(field(runs, ('cost', 'epoch_seconds_mean')), '{:.1f}'),
                     ms(field(runs, ('cost', 'gpu_hours')), '{:.2f}'),
                     ms(upd, '{:.2f}', 100), ms(overhead, '{:.2f}', 100)])
    write('cost.tex', table(
        'Training cost on one GPU (CIFAR-10, ResNet-18, 200 epochs, 90\\% target). Topology-update time '
        'includes score computation, prune/regrow and mask bookkeeping; entropy time is the part spent in '
        'the histogram hooks.', 'tab:cost', 'lcccc',
        'Method & s/epoch & GPU-hours & Topology updates (\\% of time) & Entropy histograms (\\% of time)',
        rows, SEED_NOTE))


def sensitivity_table():
    rows = []
    base = collect([plan.cfg('degst', 0.9, s) for s in plan.SEEDS])
    for v in base.values():
        rows.append(['default (bins 30, $\\Delta T$ 400, $\\alpha$ 0.2, mean)', ms([r['final_acc'] for r in v['runs']])])
    for o in plan.SENSITIVITY:
        g = collect([plan.cfg('degst', 0.9, s, **o) for s in plan.ONE_SEED])
        for v in g.values():
            k, val = next(iter(o.items()))
            lab = {'bins': 'bins', 'update_interval': '$\\Delta T$', 'prune_ratio': '$\\alpha$',
                   'entropy_reduction': 'reduction'}[k]
            rows.append([f'{lab} = {val}', ms([r['final_acc'] for r in v['runs']])])
    write('sensitivity.tex', table(
        'Sensitivity of DEGST (CIFAR-10, ResNet-18, 90\\%) to one hyper-parameter at a time. The default '
        'row has three seeds; each variation has one seed and should be read against the default row\'s std.',
        'tab:sensitivity', 'lc', 'Setting & Acc (\\%)', rows, SEED_NOTE))


def criteria_table():
    rows = []
    for sp in plan.CRITERIA_SPARSITIES:
        g = collect([plan.cfg(m, sp, s) for m in plan.CRITERIA_METHODS for s in plan.SEEDS])
        for v in g.values():
            c, runs = v['cfg'], v['runs']
            rows.append([f'{round(sp * 100)}\\%', NAMES[c.method], ms([r['final_acc'] for r in runs]),
                         ms(field(runs, ('final', 'flops_reduction_sparse')), '{:.1f}', 100)])
    write('criteria.tex', table(
        'Importance criteria inside one prune/regrow loop (CIFAR-10, ResNet-18).', 'tab:criteria', 'llcc',
        'Target & Criterion & Acc (\\%) & FLOPs red.\\ sparse (\\%)', rows, SEED_NOTE))


def legacy_table():
    rows = []
    for sp in plan.LEGACY_SPARSITIES:
        g = collect([plan.cfg(m, sp, s) for m in plan.LEGACY_METHODS for s in plan.ONE_SEED])
        for v in g.values():
            c, runs = v['cfg'], v['runs']
            rows.append([f'{round(sp * 100)}\\%', NAMES[c.method], ms([r['final_acc'] for r in runs]),
                         ms(field(runs, ('final', 'sparsity_actual')), '{:.2f}', 100),
                         ms(field(runs, ('final', 'flops_reduction_sparse')), '{:.1f}', 100)])
    write('legacy.tex', table(
        'Pre-revision pipelines re-run with the current code (CIFAR-10, ResNet-18): the first DEGST '
        'implementation (ELTP score) and one-shot pruning after 50 dense epochs with lottery-ticket rewind '
        'and 30 fine-tuning epochs.', 'tab:legacy', 'llccc',
        'Target & Method & Acc (\\%) & Sparsity (\\%) & FLOPs red.\\ sparse (\\%)', rows, SEED_NOTE))


def ext_table():
    rows = []
    for d, a in plan.EXT_SETTINGS:
        g = collect([plan.cfg(m, sp, s, dataset=d, model=a) for sp in plan.EXT_SPARSITIES
                     for m in plan.EXT_METHODS for s in plan.SEEDS])
        dense = collect([plan.cfg('dense', 0.0, s, dataset=d, model=a) for s in plan.SEEDS])
        for v in dense.values():
            rows.append([f'{d} / {a}', '--', 'Dense', ms([r['final_acc'] for r in v['runs']]), '0.0'])
        for v in g.values():
            c, runs = v['cfg'], v['runs']
            rows.append([f'{d} / {a}', f'{round(c.sparsity * 100)}\\%', NAMES[c.method],
                         ms([r['final_acc'] for r in runs]),
                         ms(field(runs, ('final', 'flops_reduction_sparse')), '{:.1f}', 100)])
    write('extension.tex', table(
        'Additional datasets and backbones.', 'tab:ext', 'lllcc',
        'Setting & Target & Method & Acc (\\%) & FLOPs red.\\ sparse (\\%)', rows, SEED_NOTE))


# --------------------------------------------------------------------------- criteria study (route 1)
DATASET_NAMES = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100', 'tinyimagenet': 'Tiny ImageNet'}
MODEL_NAMES = {'resnet18': 'ResNet-18', 'vgg16': 'VGG-16', 'mobilenetv2': 'MobileNetV2'}
# equivalence margins for TOST, fixed in experiments/DESIGN.md before the study's results
EQ_MARGIN = {'cifar10': 0.3, 'cifar100': 0.5, 'tinyimagenet': 0.5}
ALPHA = 0.05


def crit_key(method):
    """Criterion id of a study method; the robust-entropy variant gets its own id."""
    spec = METHODS[method]
    return spec['criterion'] + ('_robust' if spec.get('entropy_estimator') == 'robust' else '')


CRIT_NAMES = {'magnitude': '$|w|$', 'entropy': '$H(A)$', 'entropy_magnitude': '$H(A)\\cdot|w|$',
              'entropy_magnitude_robust': '$H_{\\mathrm{rob}}(A)\\cdot|w|$ (exploratory)',
              'taylor': 'Taylor $|wg|$', 'fisher': 'Fisher $(wg)^2/2$', 'gradient': 'Gradient $|g|$',
              'random': 'Random (control)'}
STUDY_NOTE = ('Mean $\\pm$ std over the number of seeds given in the last column; $^{*}$ marks single-seed cells '
              '(no std). Accuracy is final-epoch test accuracy.')


def diff_note(datasets):
    """Footnote of the study tables; lists the TOST margins of the datasets actually shown."""
    margins = ', '.join(f'$\\pm${EQ_MARGIN[d]} pp on {DATASET_NAMES[d]}' for d in dict.fromkeys(datasets)
                        if d in EQ_MARGIN)
    return (' $\\Delta$: difference of means to the $|w|$ row of the same block, with its 95\\% Welch CI; '
            '$p_{\\mathrm{Holm}}$: two-sided Welch test, Holm-adjusted over all confirmatory comparisons of '
            f'the table; Equiv.: TOST at {margins}, $\\alpha$ = 0.05. Exploratory rows are not in the Holm '
            'family (unadjusted $p$).')


def regime_key(d, a, sp):
    return f'{d}_{a}_s{round(sp * 100)}'


def regime_label(d, a, sp):
    return f'{DATASET_NAMES.get(d, d)}, {MODEL_NAMES.get(a, a)}, {round(sp * 100)}\\%'


def compare_to(ref, vals, margin=None):
    """Difference of means (vals - ref): Welch test and CI, TOST against ``margin``, and the C0 screen."""
    na, nb = len(ref), len(vals)
    if na < 2 or nb < 2:
        return None
    from scipy import stats as sps
    va, vb = statistics.variance(ref), statistics.variance(vals)
    diff = statistics.fmean(vals) - statistics.fmean(ref)
    pooled = math.sqrt((va + vb) / 2)
    se = math.sqrt(va / na + vb / nb)
    out = {'diff': diff, 'pooled_std': pooled, 'threshold': 1.96 * pooled, 'differs': abs(diff) > 1.96 * pooled,
           'n_ref': na, 'n': nb, 'margin': margin}
    if se == 0:          # identical values in both arms
        out.update(se=0.0, df=None, welch_t=None, p=0.0 if diff else 1.0, ci95=[diff, diff],
                   p_tost=None if margin is None else (0.0 if abs(diff) < margin else 1.0))
    else:
        df = se ** 4 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        t = diff / se
        h = sps.t.ppf(0.975, df) * se
        out.update(se=se, df=df, welch_t=t, p=float(2 * sps.t.sf(abs(t), df)), ci95=[diff - h, diff + h])
        if margin is not None:
            p_low = sps.t.sf((diff + margin) / se, df)      # H0: diff <= -margin
            p_up = sps.t.cdf((diff - margin) / se, df)      # H0: diff >= +margin
            out['p_tost'] = float(max(p_low, p_up))
    out['equivalent'] = None if margin is None or out.get('p_tost') is None else out['p_tost'] < ALPHA
    return out


def holm(pvals):
    """Holm step-down adjustment of {key: p}; returns {key: adjusted p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def _criteria_block(cfgs, margin=None):
    """Per-method summary of one block of in-loop criteria (reference: |w|)."""
    g = collect(cfgs)
    accs = {v['cfg'].method: [r['final_acc'] for r in v['runs']] for v in g.values()}
    ref = accs.get('degst_magnitude', [])
    out = {}
    for v in g.values():
        m, runs = v['cfg'].method, v['runs']
        if not runs:                  # a cell of the plan that was not run (e.g. H|w| on Tiny ImageNet)
            continue
        fl = field(runs, ('final', 'flops_reduction_sparse'))
        out[m] = {'criterion': crit_key(m), 'exploratory': m in plan.EXPLORATORY, 'accs': accs[m],
                  'n': len(accs[m]), 'mean': mean(accs[m]), 'std': stdev(accs[m]), 'flops': fl,
                  'flops_reduction_sparse': mean(fl), 'sparsity': field(runs, ('final', 'sparsity_actual')),
                  'vs_magnitude': compare_to(ref, accs[m], margin) if m != 'degst_magnitude' else None}
    return out


def _adjust(blocks):
    """Holm over the confirmatory comparisons of all blocks; stores p_holm in place."""
    ps = {(b, m): info['vs_magnitude']['p'] for b, methods in blocks.items() for m, info in methods.items()
          if info['vs_magnitude'] is not None and not info['exploratory']}
    adj = holm(ps)
    for (b, m), p in adj.items():
        blocks[b][m]['vs_magnitude']['p_holm'] = p


def _row(label, info):
    st = info['vs_magnitude']
    if st is None and info['criterion'] == 'magnitude':
        delta, p_cell, eq = '\\emph{reference}', '', ''
    elif st is None:
        delta = p_cell = eq = '--'
    else:
        delta = f'{st["diff"]:+.2f} [{st["ci95"][0]:+.2f}, {st["ci95"][1]:+.2f}]'
        p = st.get('p_holm', st['p'])
        p_cell = ('$<$0.001' if p < 1e-3 else f'{p:.3f}') + ('$^{e}$' if info['exploratory'] else '')
        eq = '--' if st['equivalent'] is None else ('yes' if st['equivalent'] else 'no')
    return [label, CRIT_NAMES[info['criterion']], ms(info['accs']), delta, p_cell, eq,
            ms(info['flops'], '{:.1f}', 100), str(info['n'])]


TABLE_HEAD = ('Regime & Drop criterion & Acc (\\%) & $\\Delta$ vs $|w|$ [95\\% CI] (pp) & $p_{\\mathrm{Holm}}$ & '
              'Equiv. & FLOPs red.\\ sparse (\\%) & Seeds')


def regimes_table():
    blocks, meta = {}, {}
    for r in plan.ALL_REGIMES:
        d, a, sp = r
        key = regime_key(d, a, sp)
        cfgs = [plan.regime_cfg(m, r, s) for m in plan.methods_for(r) for s in plan.SEEDS]
        cfgs += [plan.regime_cfg(m, r, s) for m in plan.EQUIV_METHODS for s in plan.EXTRA_SEEDS]
        block = _criteria_block(cfgs, EQ_MARGIN.get(d))
        if not any(info['n'] for info in block.values()):
            print(f'{key}: no runs yet, left out of the table')
            continue
        blocks[key] = block
        dense = collect([plan.dense_cfg(d, a, s) for s in plan.SEEDS])
        meta[key] = {'dataset': d, 'model': a, 'sparsity': sp, 'label': regime_label(d, a, sp),
                     'dense_acc': [r_['final_acc'] for v in dense.values() for r_ in v['runs']]}
    _adjust(blocks)
    rows, summary = [], {}
    for key, methods in blocks.items():
        lab = meta[key]['label']
        dacc = meta[key]['dense_acc']
        rows.append([lab, 'Dense', ms(dacc), '', '', '', '0.0', str(len(dacc))])
        rows += [_row(lab, info) for info in methods.values()]
        summary[key] = {**{k: v for k, v in meta[key].items() if k != 'label'}, 'dense_acc': mean(dacc),
                        'margin': EQ_MARGIN.get(meta[key]['dataset']), 'methods': methods}
    with open(os.path.join(plan.RESULTS, 'regimes_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    write('regimes.tex', table(
        'Drop criteria inside one prune/regrow loop across regimes (global allocation). Only the drop score '
        'differs within a block; regrowth is by gradient magnitude in every row.', 'tab:regimes', 'llcccccc',
        TABLE_HEAD, rows,
        STUDY_NOTE + diff_note(meta[k]['dataset'] for k in blocks) + ' $^{e}$ exploratory.'))
    for key, s in summary.items():
        for m, info in s['methods'].items():
            st = info['vs_magnitude']
            if st is not None:
                eq = '' if st['equivalent'] is None else (', equivalent' if st['equivalent'] else ', not equivalent')
                print(f'{key:28s} {info["criterion"]:26s} {st["diff"]:+.2f} pp '
                      f'[{st["ci95"][0]:+.2f}, {st["ci95"][1]:+.2f}] p_holm={st.get("p_holm", st["p"]):.3g}{eq} '
                      f'(n={info["n"]})')
    return summary


def allocation_table():
    """A/B/C with global vs layer-uniform allocation (CIFAR-10, ResNet-18, 90%)."""
    blocks = {}
    for alloc, kw in (('global', {}), ('uniform', {'allocation': 'uniform'})):
        blocks[alloc] = _criteria_block([plan.cfg(m, 0.9, s, **kw) for m in plan.C0_METHODS for s in plan.SEEDS],
                                        EQ_MARGIN['cifar10'])
    _adjust(blocks)
    rows = [_row(alloc, info) for alloc, methods in blocks.items() for info in methods.values()]
    with open(os.path.join(plan.RESULTS, 'allocation_summary.json'), 'w') as f:
        json.dump(blocks, f, indent=1)
    # measured sparsity counts exact zeros: a weight regrown where the gradient is zero (e.g. it reads a
    # dead channel) stays zero, which only happens when every layer must regrow its own quota (uniform)
    off = [(alloc, CRIT_NAMES[info['criterion']], min(info['sparsity']), max(info['sparsity']))
           for alloc, methods in blocks.items() for info in methods.values()
           if info['sparsity'] and max(abs(s - 0.9) for s in info['sparsity']) > 1e-3]
    note = ''.join(f' Measured sparsity of {lab} ({alloc}): {100 * lo:.2f}--{100 * hi:.2f}\\%, because weights '
                   f'regrown where the gradient is exactly zero (inputs from dead channels) stay zero.'
                   for alloc, lab, lo, hi in off)
    write('allocation.tex', table(
        'C0 criteria with global and layer-uniform allocation (CIFAR-10, ResNet-18, 90\\%). With uniform '
        'allocation every layer keeps the same density, so differences can only come from which weights '
        'are dropped inside a layer.', 'tab:allocation', 'llcccccc',
        TABLE_HEAD.replace('Regime', 'Allocation'), rows, STUDY_NOTE + diff_note(['cifar10']) + note))


def entropy_vs_random_table(summary):
    """Post hoc (not in DESIGN.md before the results): H alone against the random-drop control."""
    rows, out, ps = [], {}, {}
    for key, s in summary.items():
        ms_ = s['methods']
        if 'degst_entropy' not in ms_ or 'dst_random' not in ms_:
            continue
        st = compare_to(ms_['dst_random']['accs'], ms_['degst_entropy']['accs'])
        if st is None:
            continue
        out[key] = st
        ps[key] = st['p']
    for key, p in holm(ps).items():
        out[key]['p_holm'] = p
    for key, st in out.items():
        s = summary[key]
        lab = regime_label(s['dataset'], s['model'], s['sparsity'])
        p = st['p_holm']
        rows.append([lab, ms(s['methods']['degst_entropy']['accs']), ms(s['methods']['dst_random']['accs']),
                     f'{st["diff"]:+.2f} [{st["ci95"][0]:+.2f}, {st["ci95"][1]:+.2f}]',
                     '$<$0.001' if p < 1e-3 else f'{p:.3f}'])
    with open(os.path.join(plan.RESULTS, 'entropy_vs_random.json'), 'w') as f:
        json.dump(out, f, indent=1)
    write('entropy_vs_random.tex', table(
        'Post hoc comparison: drop by entropy alone against random drop, same loop and regrowth. Both drop sets '
        'have a Jaccard overlap below 0.1 with that of $|w|$ (Fig.~\\ref{fig:divergence}).', 'tab:hrandom', 'lcccc',
        'Regime & $H(A)$ (\\%) & Random (\\%) & $\\Delta$ $H(A) -$ random [95\\% CI] (pp) & $p_{\\mathrm{Holm}}$',
        rows, 'Welch tests, Holm-adjusted over the regimes shown. This comparison was not planned before the '
        'results; it is reported as a post hoc analysis.'))
    for key, st in out.items():
        print(f'post hoc H - random {key:24s} {st["diff"]:+.2f} [{st["ci95"][0]:+.2f}, {st["ci95"][1]:+.2f}] '
              f'p_holm={st["p_holm"]:.3g}')


# --------------------------------------------------------------------------- figures of the criteria study
CRIT_MARKERS = {'entropy': 's', 'entropy_magnitude': 'o', 'entropy_magnitude_robust': 'P', 'taylor': '^',
                'fisher': 'v', 'gradient': 'D', 'random': 'X', 'magnitude': '.'}
SHORT_NAMES = {'magnitude': '$|w|$', 'entropy': '$H$', 'entropy_magnitude': '$H\\cdot|w|$',
               'entropy_magnitude_robust': '$H_{rob}\\cdot|w|$*', 'taylor': 'Taylor', 'fisher': 'Fisher',
               'gradient': '$|g|$', 'random': 'random'}


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, name)
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')


def penalty_figure(summary):
    """Accuracy change vs |w| (95% CI) per criterion, one column per regime, with the TOST margin band."""
    plt = _plt()
    keys = [k for k in summary if any(i['n'] for i in summary[k]['methods'].values())]
    crits = [c for c in CRIT_MARKERS if c != 'magnitude']
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    width = 0.8 / len(crits)
    for x, key in enumerate(keys):
        s = summary[key]
        m = s.get('margin')
        if m:
            ax.fill_between([x - 0.45, x + 0.45], -m, m, color='grey', alpha=.15, lw=0)
        by_crit = {i['criterion']: i for i in s['methods'].values()}
        for j, c in enumerate(crits):
            info = by_crit.get(c)
            st = info and info['vs_magnitude']
            if not st:
                continue
            xx = x - 0.4 + (j + 0.5) * width
            ax.errorbar(xx, st['diff'], yerr=[[st['diff'] - st['ci95'][0]], [st['ci95'][1] - st['diff']]],
                        marker=CRIT_MARKERS[c], color=f'C{j}', ms=5, capsize=2, lw=1,
                        mfc='none' if info['exploratory'] else f'C{j}', label=SHORT_NAMES[c] if x == 0 else None)
    ax.axhline(0, color='k', lw=.8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([regime_label(summary[k]['dataset'], summary[k]['model'], summary[k]['sparsity'])
                        .replace('\\%', '%').replace(', ', '\n', 1) for k in keys], fontsize=8)
    ax.set_ylabel('accuracy change vs $|w|$ (pp)')
    ax.grid(axis='y', alpha=.3)
    handles, labels = ax.get_legend_handles_labels()
    # criteria absent from the first regime still need a legend entry
    for j, c in enumerate(crits):
        if SHORT_NAMES[c] not in labels:
            handles.append(plt.Line2D([], [], marker=CRIT_MARKERS[c], color=f'C{j}', ls=''))
            labels.append(SHORT_NAMES[c])
    ax.legend(handles, labels, fontsize=7, ncol=4, loc='lower left')
    _save(fig, 'penalty_by_regime.pdf')
    plt.close(fig)


def entropy_sparsity_figure(name='cifar100_resnet18_degst_magnitude_s90_seed0_ep50'):
    """Per-channel entropy against the share of exact zeros in the channel's input (replay checkpoint).

    Per-channel values are read from results/analysis_channels/ (a separate replay of this one
    checkpoint, so the H4 files in results/analysis/ stay as they are), else from results/analysis/.
    """
    candidates = [os.path.join(plan.RESULTS, sub, f'overlap_{name}.json') for sub in ('analysis_channels', 'analysis')]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        print(f'entropy_sparsity_figure: overlap_{name}.json missing, skipped')
        return
    with open(path) as f:
        o = json.load(f)
    ch = o.get('channels')
    if not ch:
        print('entropy_sparsity_figure: no per-channel data (re-run analysis/score_overlap.py), skipped')
        return
    plt = _plt()
    layers = list(ch)[1:]                       # the first layer reads the image (no ReLU zeros)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    cmap = plt.get_cmap('viridis')
    for i, n in enumerate(layers):
        col = cmap(i / max(1, len(layers) - 1))
        z = ch[n]['zero']
        axes[0].scatter(z, ch[n]['H'], s=3, color=col, alpha=.5, lw=0)
        axes[1].scatter(z, ch[n]['H_robust'], s=3, color=col, alpha=.5, lw=0)
    rho = o['summary'].get('spearman_H_zero_pooled')
    axes[0].set_title(f'histogram entropy $H$ (Spearman {rho:+.2f})' if rho is not None else '$H$', fontsize=9)
    axes[1].set_title('robust entropy $H_{rob}$ (non-zero values)', fontsize=9)
    for ax in axes:
        ax.set_xlabel('share of exact zeros in the channel input')
        ax.grid(alpha=.3)
    axes[0].set_ylabel('entropy (nats)')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, len(layers)))
    fig.colorbar(sm, ax=axes, label='layer (input side)', fraction=.03)
    _save(fig, 'entropy_vs_sparsity.pdf')
    plt.close(fig)


CURVE_METHODS = ('degst_magnitude', 'degst', 'dst_random', 'degst_entropy')


def learning_curves_figure():
    """Post hoc: train (every epoch) and test (every eval) accuracy over training, per regime.

    The x-axis is the fraction of training (iterations), so Tiny ImageNet (100 epochs) and CIFAR
    (200 epochs) line up; dotted lines mark the end of the warm-up and T_end. Also writes the
    final train/test accuracies to results/learning_curves.json.
    """
    plt = _plt()
    curves, final = {}, {}
    for r in plan.ALL_REGIMES:
        key = regime_key(*r)
        for m in CURVE_METHODS:
            if m not in plan.methods_for(r):
                continue
            hs = [load(plan.regime_cfg(m, r, s)) for s in plan.SEEDS]
            hs = [x['history'] for x in hs if x]
            if not hs:
                continue
            n = len(hs[0])
            tr = [mean([h[e]['train_acc'] for h in hs]) for e in range(n)]
            te = [(e, mean([h[e].get('test_acc') for h in hs])) for e in range(n)
                  if hs[0][e].get('test_acc') is not None]
            curves.setdefault(key, {})[m] = (n, tr, te)
            final.setdefault(key, {})[m] = {'train_acc': tr[-1], 'test_acc': te[-1][1], 'n': len(hs)}
    if not curves:
        print('learning_curves_figure: no runs, skipped')
        return
    with open(os.path.join(plan.RESULTS, 'learning_curves.json'), 'w') as f:
        json.dump(final, f, indent=1)
    keys = list(curves)
    # drawn at the printed width (7.2 in), so fonts are not scaled down by LaTeX
    fig, axes = plt.subplots(2, len(keys), figsize=(7.2, 4.0), squeeze=False)
    for j, key in enumerate(keys):
        regime = next(r for r in plan.ALL_REGIMES if regime_key(*r) == key)
        c = plan.regime_cfg(next(iter(curves[key])), regime, 0)
        for m, (n, tr, te) in curves[key].items():
            col = f'C{CURVE_METHODS.index(m)}'
            lab = SHORT_NAMES[crit_key(m)]
            axes[0][j].plot([(e + 1) / n for e in range(n)], tr, color=col, lw=1.1, label=lab)
            axes[1][j].plot([(e + 1) / n for e, _ in te], [v for _, v in te], color=col, lw=1.1, marker='.', ms=3)
        for ax in (axes[0][j], axes[1][j]):
            for x in (c.warmup_epochs / c.epochs, c.end_epoch / c.epochs):
                ax.axvline(x, color='grey', ls=':', lw=.8)
            ax.grid(alpha=.3)
            ax.tick_params(labelsize=6)
            ax.set_xticks([0, 0.5, 1])
        axes[0][j].set_title(regime_label(*regime).replace('\\%', '%').replace(', ', '\n', 1), fontsize=7)
        axes[1][j].set_xlabel('fraction of training', fontsize=7)
    axes[0][0].set_ylabel('train accuracy (%)', fontsize=7)
    axes[1][0].set_ylabel('test accuracy (%)', fontsize=7)
    axes[0][0].legend(fontsize=6, loc='lower right')
    fig.tight_layout()
    _save(fig, 'learning_curves.pdf')
    plt.close(fig)


def jaccard_figure(keys=('cifar10_resnet18_s90', 'cifar100_resnet18_s90')):
    """Jaccard overlap between the drop sets of every pair of criteria (mean over the replay checkpoints)."""
    import glob
    order = ['magnitude', 'entropy_magnitude_robust', 'entropy_magnitude', 'taylor', 'fisher', 'gradient',
             'entropy', 'random']
    mats = {}
    for path in glob.glob(os.path.join(plan.RESULTS, 'analysis', 'overlap_*.json')):
        with open(path) as f:
            o = json.load(f)
        if o.get('method') != 'degst_magnitude' or o.get('epoch') is None or 'sparsity_pct' not in o:
            continue
        key = f'{o["dataset"]}_{o["model"]}_s{o["sparsity_pct"]}'
        if key in keys:
            mats.setdefault(key, []).append(o['summary']['jaccard'])
    keys = [k for k in keys if k in mats]
    if not keys:
        print('jaccard_figure: no replay files, skipped')
        return
    plt = _plt()
    fig, axes = plt.subplots(1, len(keys), figsize=(7.2, 3.4), squeeze=False)
    fig.subplots_adjust(wspace=0.55)            # room for the y labels of the right panel
    for ax, key in zip(axes[0], keys):
        crit = [c for c in order if all(c in J for J in mats[key])]
        m = [[mean([J[a][b] for J in mats[key]]) for b in crit] for a in crit]
        im = ax.imshow(m, vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(len(crit)))
        ax.set_yticks(range(len(crit)))
        ax.set_xticklabels([SHORT_NAMES[c] for c in crit], rotation=60, fontsize=6.5)
        ax.set_yticklabels([SHORT_NAMES[c] for c in crit], fontsize=6.5)
        for i in range(len(crit)):
            for j in range(len(crit)):
                ax.text(j, i, f'{m[i][j]:.2f}', ha='center', va='center', fontsize=5.5,
                        color='black' if m[i][j] > 0.5 else 'white')
        d, a, sp = key.split('_')
        ax.set_title(regime_label(d, a, int(sp[1:]) / 100).replace('\\%', '%'), fontsize=8)
    fig.colorbar(im, ax=list(axes[0]), fraction=.025, label='Jaccard index of drop sets')
    _save(fig, 'jaccard_matrix.pdf')
    plt.close(fig)


def layer_density_figure(key='cifar10_resnet18_s90'):
    """Per-layer density of the final masks and each layer's share of the dense FLOPs."""
    path = os.path.join(plan.RESULTS, 'analysis', 'layer_density.json')
    if not os.path.exists(path):
        print(f'layer_density_figure: {path} missing (run analysis/layer_density.py), skipped')
        return
    with open(path) as f:
        data = json.load(f)
    if key not in data:
        print(f'layer_density_figure: {key} not in {path}, skipped')
        return
    d = data[key]
    plt = _plt()
    x = range(len(d['layers']))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for method, info in d['methods'].items():
        c = crit_key(method)
        if c not in ('magnitude', 'entropy', 'entropy_magnitude', 'random', 'taylor'):
            continue
        ax.plot(x, info['density'], marker=CRIT_MARKERS[c], ms=4, lw=1.2, label=SHORT_NAMES[c])
    flops = [p * n for p, n in zip(d['positions'], d['numel'])]
    tot = sum(flops)
    ax2 = ax.twinx()
    ax2.bar(x, [100 * f / tot for f in flops], color='grey', alpha=.2, width=.8)
    ax2.set_ylabel('share of dense FLOPs (%)', color='grey')
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_yscale('log')
    ax.set_ylabel('density of final mask')
    ax.set_xticks(list(x))
    ax.set_xticklabels([n.replace('layer', 'L').replace('.conv', '.c').replace('shortcut.0', 'sc')
                        for n in d['layers']], rotation=90, fontsize=6)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc='upper right')
    _save(fig, f'layer_density_{key}.pdf')
    plt.close(fig)


def divergence_figure(summary):
    """Accuracy change vs how far a criterion's drop set is from |w|'s (mid-training checkpoints)."""
    import glob

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    div = {}
    for path in glob.glob(os.path.join(plan.RESULTS, 'analysis', 'overlap_*.json')):
        with open(path) as f:
            o = json.load(f)
        if o.get('method') != 'degst_magnitude' or o.get('epoch') is None or 'sparsity_pct' not in o:
            continue
        key = f'{o["dataset"]}_{o["model"]}_s{o["sparsity_pct"]}'
        for c, j in o['summary']['jaccard']['magnitude'].items():
            div.setdefault(key, {}).setdefault(c, []).append(1 - j)

    markers = {'entropy': 's', 'entropy_magnitude': 'o', 'entropy_magnitude_robust': 'P', 'taylor': '^',
               'fisher': 'v', 'gradient': 'D', 'random': 'X'}
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    points, used = [], []
    for i, (d, a, sp) in enumerate(plan.ALL_REGIMES):
        key = regime_key(d, a, sp)
        if key not in div or key not in summary:
            continue
        used.append((key, colors[i % len(colors)], regime_label(d, a, sp).replace('\\%', '%')))
        for m, info in summary[key]['methods'].items():
            c, st = info['criterion'], info['vs_magnitude']
            if st is None or c not in div[key] or c not in markers:
                continue
            x = mean(div[key][c])
            points.append({'regime': key, 'criterion': c, 'exploratory': info['exploratory'], 'divergence': x,
                           'diff': st['diff'], 'ci95': st['ci95']})
            ax.errorbar(x, st['diff'], yerr=[[st['diff'] - st['ci95'][0]], [st['ci95'][1] - st['diff']]],
                        marker=markers[c], color=colors[i % len(colors)], ms=6, capsize=2, lw=1,
                        mfc='none' if info['exploratory'] else colors[i % len(colors)])
    ax.axhline(0, color='grey', lw=.8)
    ax.set_xlabel('drop-set divergence from $|w|$ (1 - Jaccard)')
    ax.set_ylabel('accuracy change vs $|w|$ (pp)')
    ax.grid(alpha=.3)
    if points:
        h1 = [Line2D([], [], color=col, marker='o', ls='', label=lab) for _, col, lab in used]
        h2 = [Line2D([], [], color='k', marker=mk, ls='',
                     label=CRIT_NAMES[c].replace(' (DEGST)', '').replace(' (exploratory)', '*'))
              for c, mk in markers.items()]
        ax.legend(handles=h1 + h2, fontsize=7, loc='lower left')
    else:
        ax.text(0.5, 0.5, 'results pending', ha='center', va='center', transform=ax.transAxes)
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, 'divergence_accuracy.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')
    # H4 is tested on the confirmatory points only
    conf = [p for p in points if not p['exploratory']]
    rho = pval = None
    if len(conf) > 2:
        from scipy import stats as sps
        res = sps.spearmanr([p['divergence'] for p in conf], [p['diff'] for p in conf])
        rho, pval = float(res[0]), float(res[1])
    # post hoc robustness (not planned): the divergences fall into two clusters, and the points share
    # regimes and the |w| reference, so the pooled p-value is optimistic
    post = {}
    if len(conf) > 2:
        from scipy import stats as sps
        within = {}
        for key in dict.fromkeys(p['regime'] for p in conf):
            q = [p for p in conf if p['regime'] == key]
            if len(q) > 2:
                within[key] = float(sps.spearmanr([p['divergence'] for p in q], [p['diff'] for p in q])[0])
        near = [p for p in conf if p['divergence'] < 0.5]
        far = [p for p in conf if p['divergence'] >= 0.5]
        post = {'spearman_within_regime': within,
                'near': {'n': len(near), 'divergence': [min((p['divergence'] for p in near), default=None),
                                                        max((p['divergence'] for p in near), default=None)],
                         'diff': [min((p['diff'] for p in near), default=None),
                                  max((p['diff'] for p in near), default=None)]},
                'far': {'n': len(far), 'divergence': [min((p['divergence'] for p in far), default=None),
                                                      max((p['divergence'] for p in far), default=None)],
                        'diff': [min((p['diff'] for p in far), default=None),
                                 max((p['diff'] for p in far), default=None)]}}
    with open(os.path.join(plan.RESULTS, 'divergence.json'), 'w') as f:
        json.dump({'spearman_divergence_vs_diff': rho, 'p_value': pval, 'n_points': len(conf),
                   'post_hoc': post, 'points': points}, f, indent=1)
    if rho is not None:
        print(f'divergence vs accuracy change: Spearman {rho:+.2f} (p = {pval:.3g}) over {len(conf)} points')
        print(f'  post hoc: within-regime rho {post["spearman_within_regime"]}; near {post["near"]}; far {post["far"]}')


# --------------------------------------------------------------------------- Pareto
def pareto(dataset='cifar10', model='resnet18'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfgs = [plan.cfg(m, sp, s, dataset=dataset, model=model) for m in plan.MAIN_METHODS
            for sp in plan.PARETO_SPARSITIES for s in plan.pareto_seeds(sp)]
    g = collect(cfgs)
    pts = {}
    for v in g.values():
        runs = v['runs']
        if not runs:
            continue
        m = v['cfg'].method
        pts.setdefault(m, []).append(dict(
            sp=v['cfg'].sparsity, acc=mean([r['final_acc'] for r in runs]),
            acc_sd=stdev([r['final_acc'] for r in runs]) or 0.0,
            fs=mean([r['final']['flops_sparse'] / r['final']['dense_flops'] for r in runs]),
            fd=mean([r['final']['flops_dense'] / r['final']['dense_flops'] for r in runs])))
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, f'pareto_{dataset}_{model}.pdf')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, key, title in zip(axes, ('fs', 'fd'), ('non-zero FLOPs (sparse)', 'executed FLOPs (dense kernels)')):
        for m, p in sorted(pts.items()):
            p = sorted(p, key=lambda d: d[key])
            ls = '-' if KIND.get(m) == 'U' else ('--' if KIND.get(m) == 'S' else ':')
            ax.errorbar([d[key] * 100 for d in p], [d['acc'] for d in p], yerr=[d['acc_sd'] for d in p],
                        marker='o', ms=3, lw=1.2, ls=ls, capsize=2, label=NAMES[m].replace('$^\\dagger$', '*'))
        ax.set_xscale('log')
        ax.set_xlabel(f'{title}, % of dense')
        ax.grid(alpha=.3)
    axes[0].set_ylabel('test accuracy (%)')
    if not pts:
        for ax in axes:
            ax.text(0.5, 0.5, 'results pending', ha='center', va='center', transform=ax.transAxes)
    else:
        axes[1].legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')

    # dominance on the sparse-FLOPs axis: a point is dominated if another method is at least as
    # accurate with at most as many FLOPs, strictly better in one of the two
    flat = [(m, d) for m, p in pts.items() for d in p]
    dominated = []
    for m, d in flat:
        for m2, d2 in flat:
            if m2 != m and d2['acc'] >= d['acc'] and d2['fs'] <= d['fs'] and (d2['acc'] > d['acc'] or d2['fs'] < d['fs']):
                dominated.append({'method': m, 'sparsity': d['sp'], 'by': m2, 'by_sparsity': d2['sp']})
                break
    with open(os.path.join(plan.RESULTS, f'pareto_dominance_{dataset}_{model}.json'), 'w') as f:
        json.dump(dominated, f, indent=1)
    for x in dominated:
        if x['method'] in ('degst', 'degst_s', 'hybrid'):
            print(f"dominated: {x['method']}@{x['sparsity']} by {x['by']}@{x['by_sparsity']}")


def main():
    """Tables and figures of the criteria study; ``--all`` also writes those of the pre-C0 plan."""
    import sys
    os.makedirs(plan.RESULTS, exist_ok=True)
    c0()
    summary = regimes_table()
    allocation_table()
    divergence_figure(summary)
    entropy_vs_random_table(summary)
    penalty_figure(summary)
    entropy_sparsity_figure()
    layer_density_figure()
    learning_curves_figure()
    jaccard_figure()
    if '--all' in sys.argv:
        main_table()
        speed_table()
        cost_table()
        sensitivity_table()
        criteria_table()
        ext_table()
        legacy_table()
        pareto()


if __name__ == '__main__':
    main()
