"""Run configuration and the method registry.

Every method is a set of overrides on ``RunConfig``; nothing else distinguishes
them, so any difference in results comes from those overrides. The registry is
the single place to read what a method name means.
"""
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RunConfig:
    # what
    dataset: str = 'cifar10'
    model: str = 'resnet18'
    method: str = 'degst'
    sparsity: float = 0.9            # fraction of conv/linear weights removed, w.r.t. the dense model
    seed: int = 0
    # optimisation (identical for every method; no per-method tuning in the experiment plan)
    epochs: int = 200
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    amp: bool = True                 # bf16 autocast on CUDA
    # sparse-training schedule
    kind: str = 'dst'                # dense | dst | gradual | oneshot_init | posttrain | structured | hybrid
    criterion: str = 'entropy_magnitude'
    allocation: str = 'global'       # global | uniform | erk
    init: str = 'score'              # score (after warm-up) | random
    warmup_epochs: int = 10          # dense warm-up before the first mask
    update_interval: int = 400       # iterations between topology updates (Delta T)
    prune_ratio: float = 0.2         # fraction of active weights/channels swapped per update (alpha)
    ratio_decay: str = 'cosine'      # cosine | constant
    end_epoch: float = 150           # T_end: no topology changes afterwards
    min_density: float = 0.01        # per-layer floor for global selection
    # one-shot pruning after dense training (pre-revision pipeline: train -> prune -> LTH rewind -> fine-tune)
    prune_epoch: float = 50          # dense phase length; the rest of `epochs` is fine-tuning
    rewind: bool = True              # reset kept weights to their initial values (lottery ticket)
    score_at: str = 'prune'          # prune | init: when the posttrain scores are computed
    # entropy estimation
    bins: int = 30
    entropy_batches: int = 4         # training batches pooled per update
    entropy_reduction: str = 'mean'  # mean | median over those batches
    entropy_max_samples: int = 0     # per-channel subsample (0 = all values)
    entropy_estimator: str = 'histogram'  # histogram (C0) | robust (non-zero values, quantile range)
    # structured
    channel_criterion: str = 'entropy'   # entropy | bn_gamma | fpgm | l1 | learned
    channel_schedule: str = 'gradual'    # gradual | oneshot
    channel_swap: bool = True            # channel prune/regrow via |dL/dgate|
    min_channel_keep: float = 0.05
    slimming_lambda: float = 0.0
    autopruner_lambda: float = 10.0
    autopruner_alpha_max: float = 100.0
    hybrid_struct_share: float = 0.5     # structured stage removes this share of the target weights
    # evaluation / bookkeeping
    val_size: int = 0                    # >0: hold out from train and evaluate on it (tuning only)
    eval_every: int = 10                 # evaluate every n epochs (+ first and last); final acc = last epoch
    speed_batches: Tuple[int, ...] = (1, 128, 1024)
    speed_warmup: int = 50
    speed_iters: int = 200
    data_root: str = './data'
    num_workers: int = 8
    out_dir: str = 'results'
    save_checkpoint: bool = True
    checkpoint_epochs: Tuple[int, ...] = ()   # extra mid-training checkpoints (model + masks), for analysis
    tag: str = ''                        # free-form suffix, e.g. suite name
    overrides: dict = field(default_factory=dict)  # recorded for the run id

    def run_id(self) -> str:
        base = f'{self.dataset}_{self.model}_{self.method}_s{round(self.sparsity * 100)}_seed{self.seed}'
        extra = {k: v for k, v in sorted(self.overrides.items())}
        if self.val_size:
            extra['val_size'] = self.val_size
        if extra:
            base += '_' + hashlib.md5(json.dumps(extra, sort_keys=True).encode()).hexdigest()[:8]
        return base

    def to_dict(self):
        return dataclasses.asdict(self)


# name -> overrides. Unlisted fields keep the RunConfig defaults above.
METHODS = {
    'dense': dict(kind='dense'),

    # --- DEGST and the C0 ablation: identical loop, only the drop criterion changes
    'degst': dict(kind='dst', criterion='entropy_magnitude'),          # C0 config C
    'degst_entropy': dict(kind='dst', criterion='entropy'),            # C0 config B
    'degst_magnitude': dict(kind='dst', criterion='magnitude'),        # C0 config A
    # further criteria in the same loop (study of importance criteria / perturbation-aware scores)
    'dst_taylor': dict(kind='dst', criterion='taylor'),
    'dst_fisher': dict(kind='dst', criterion='fisher'),
    'dst_gradient': dict(kind='dst', criterion='gradient'),
    'dst_eltp': dict(kind='dst', criterion='eltp'),      # pre-revision score in the current loop
    'dst_random': dict(kind='dst', criterion='random'),  # control: random drop, same loop and regrowth
    # exploratory (added after the C0 diagnostics, before the study's results): DEGST with an entropy
    # estimate that ignores exact zeros and outliers (degst/entropy.py: channel_entropy_robust)
    'degst_robust': dict(kind='dst', criterion='entropy_magnitude', entropy_estimator='robust'),

    # --- unstructured baselines, each with its original paper's conventions
    'rigl': dict(kind='dst', criterion='magnitude', allocation='erk', init='random', warmup_epochs=0,
                 update_interval=100, prune_ratio=0.3, end_epoch=150),
    'gmp': dict(kind='gradual', criterion='magnitude', allocation='uniform'),
    'movement': dict(kind='gradual', criterion='movement', allocation='uniform'),
    'snip': dict(kind='oneshot_init', criterion='snip', allocation='global', warmup_epochs=0),

    # --- structured (channels removed physically with DepGraph)
    'degst_s': dict(kind='structured', channel_criterion='entropy', channel_swap=True),
    'slimming': dict(kind='structured', channel_criterion='bn_gamma', channel_schedule='oneshot',
                     channel_swap=False, slimming_lambda=1e-4),
    'fpgm': dict(kind='structured', channel_criterion='fpgm', channel_swap=False),
    'l1filter': dict(kind='structured', channel_criterion='l1', channel_swap=False),
    'autopruner': dict(kind='structured', channel_criterion='learned', channel_schedule='oneshot',
                       channel_swap=False),

    # --- structured stage then unstructured DEGST inside the surviving channels
    'hybrid': dict(kind='hybrid', channel_criterion='entropy', channel_swap=True,
                   criterion='entropy_magnitude'),

    # --- pre-revision code (archive/legacy_code.zip), kept to reproduce and compare with earlier numbers.
    # DEGST as first implemented: ELTP score, random uniform mask, no warm-up, 5% swap every 100
    # steps until the end, lr 0.01 (the old code had no lr schedule; here cosine like all methods).
    'degst_v0': dict(kind='dst', criterion='eltp', init='random', warmup_epochs=0, update_interval=100,
                     prune_ratio=0.05, ratio_decay='constant', end_epoch=200, lr=0.01),
    # train dense 50 epochs -> one-shot prune -> rewind kept weights to init -> fine-tune 30 epochs
    'posttrain_magnitude': dict(kind='posttrain', criterion='magnitude', allocation='global',
                                epochs=80, prune_epoch=50),
    'posttrain_layerwise': dict(kind='posttrain', criterion='magnitude', allocation='uniform',
                                epochs=80, prune_epoch=50),
    'posttrain_snip': dict(kind='posttrain', criterion='snip', allocation='global', epochs=80, prune_epoch=50),
    'posttrain_eltp': dict(kind='posttrain', criterion='eltp', allocation='global', epochs=80, prune_epoch=50,
                           score_at='init'),
}


def make_config(method: str, **kw) -> RunConfig:
    if method not in METHODS:
        raise KeyError(f'unknown method {method}; choose from {sorted(METHODS)}')
    types = {f.name: f.type for f in dataclasses.fields(RunConfig)}
    fields = set(types)
    base = dict(METHODS[method])
    user = {k: v for k, v in kw.items() if v is not None}
    bad = set(user) - fields
    if bad:
        raise KeyError(f'unknown config fields {bad}')
    # same value, same run id: train.py parses float fields as floats (--end-epoch 75 -> 75.0), so an int
    # given for a float field (end_epoch=75 in experiments/plan.py) must hash the same way
    user = {k: float(v) if types[k] in (float, 'float') and isinstance(v, int) and not isinstance(v, bool) else v
            for k, v in user.items()}
    identity = {'dataset', 'model', 'sparsity', 'seed', 'data_root', 'num_workers', 'out_dir',
                'save_checkpoint', 'checkpoint_epochs', 'tag', 'speed_batches', 'speed_warmup', 'speed_iters',
                'val_size', 'eval_every'}
    overrides = {k: v for k, v in user.items() if k not in identity and base.get(k, getattr(RunConfig, k, None)) != v}
    cfg = RunConfig(method=method, **{**base, **user})
    cfg.overrides = overrides
    return cfg
