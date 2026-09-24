"""Aggregate results/*.json into the paper's tables and figures.

    python aggregate.py            # writes paper/tables/*.tex, paper/Figures/pareto_*.pdf,
                                   # results/c0_decision.json, results/pareto_dominance.json

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
        rows.append([lab, NAMES[m], {'degst_magnitude': '$|w|$', 'degst_entropy': '$H(A)$',
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
                   'NEGATIVE: no benefit over |w| -> switch to route 1 (criteria study) or route 2')
        print(f'C0: C-A = {diff:+.3f} pp, threshold 1.96*pooled = {1.96 * pooled:.3f} pp  => {verdict}')
    else:
        print('C0: not enough runs for a decision', decision['n'])
    os.makedirs(plan.RESULTS, exist_ok=True)
    with open(os.path.join(plan.RESULTS, 'c0_decision.json'), 'w') as f:
        json.dump(decision, f, indent=1)
    write('c0_ablation.tex', table(
        'Ablation C0: the same prune/regrow loop with three drop criteria (CIFAR-10, ResNet-18, 90\\% '
        'target sparsity). Only the importance score differs between rows.',
        'tab:c0', 'lllccccc',
        'Cfg & Loop & Drop score & Acc (\\%) & Sparsity (\\%) & FLOPs red. (\\%) & Non-zero weights & Seeds',
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
    os.makedirs(plan.RESULTS, exist_ok=True)
    c0()
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
