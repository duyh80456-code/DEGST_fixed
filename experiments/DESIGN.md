# Experiment design after C0 (criteria study)

Written 2026-09-28, **before** any run of the suites below. Changes after that date go in the
changelog at the end, with the reason, so that the paper can report what was planned and what was not.

## What C0 showed

CIFAR-10, ResNet-18, 90%, global allocation, 3 seeds, identical loop (`results/c0_decision.json`):

| Config | Drop score | Acc (%) | vs A |
| --- | --- | --- | --- |
| A | \|w\| | 95.15 | — |
| B | H(A) | ≈ 92.71 | −2.45 pp |
| C | H(A)·\|w\| (DEGST) | 95.12 | −0.04 pp (threshold 0.31) |

Diagnostics on the final A checkpoints (`analysis/score_overlap.py`, 3 seeds, all within ±0.01):
the drop sets of A and C overlap with Jaccard ≈ 0.72 (≈ 0.76 within layers at equal counts), A and B
≈ 0.13. Entropy is not uniform (CV 0.3–1.0 per layer, dead channels with H ≈ 0), and multiplying by
it moves the budget towards early, high-resolution layers: the dropped weights carry ≈ 16.4% of the
active FLOPs under A, ≈ 13.9% under C, ≈ 11.3% under B, which is why C's final FLOPs reduction is
lower (0.71 vs 0.76).

So entropy changes *which* weights are dropped, without changing accuracy (C), and a criterion far
from |w| costs accuracy (B). The paper can no longer claim that DEGST improves on magnitude.

## Research question

> In dynamic sparse training, how much does the drop criterion matter, and when?

Operationalised as: the accuracy change of a criterion relative to |w| inside the same loop, as a
function of (i) how far its drop set is from |w|'s and (ii) the difficulty of the regime, scaled by
the gap between |w| and a random drop.

## Hypotheses and primary claims (fixed before running; revised 2026-09-28, see changelog)

- **H0 (anchor).** Random drop (same loop, gradient regrowth) is worse than |w| in every regime.
  It sets the scale for every other difference.
- **H1 (magnitude is necessary).** In every regime, H(A) alone is worse than |w|.
- **H2 (tolerance).** H·|w| is *equivalent* to |w| in every regime (TOST, margins below).
- **H3 (difficulty).** The penalty for diverging from |w| grows with difficulty: at 98% sparsity,
  on CIFAR-100, with VGG-16 and on Tiny ImageNet, more criteria differ from |w|, by more.
- **H4 (dose–response).** Across criteria and regimes, the accuracy change vs |w| decreases with the
  drop-set divergence from |w| (1 − Jaccard, measured at mid-training).
- **H5 (mechanism of B's loss).** With layer-uniform allocation (no budget shift between layers),
  B − A shrinks. If it does not, B's loss comes from the choice of weights within layers.
- **E1 (exploratory).** The plain histogram entropy after a ReLU mostly measures the share of exact
  zeros (C0 diagnostics: dead channels, CV 0.3–1.0). H_robust·|w| — entropy of the non-zero values,
  bins over the 1–99 % quantile range — tests whether a cleaner entropy estimate changes the C0
  conclusion. No tuning; reported with unadjusted p, outside the confirmatory family.

Any of these may come out negative; all outcomes are reported.

## Suites

All: 200 epochs, lr 0.1 cosine, batch 128, same loop and schedule as C0 (warm-up 10 epochs, ΔT 400,
α 0.2 cosine, T_end 150), regrowth by gradient magnitude. Sized for one GPU:

- **Main regimes** (ResNet-18 CIFAR-10 90%, ResNet-18 CIFAR-100 90%): all 8 criteria — |w|, H,
  H·|w|, random, Taylor, Fisher, |g|, H_robust·|w|.
- **Secondary regimes** (ResNet-18 CIFAR-10 98%, VGG-16 CIFAR-100 90%; optional Tiny ImageNet 90%):
  the key criteria |w|, H, H·|w|, random.

| Suite | Content | Runs (new) | Answers |
| --- | --- | --- | --- |
| `regimes` | criteria × regimes as above, 3 seeds; the \|w\| seed-0 run of each regime also saves mid-training checkpoints (epochs 50, 100) | 72 (63) | H0–H3, E1 |
| `power` | seeds 3, 4 for \|w\| and H·\|w\| in every regime (5 seeds for the equivalence claim) | 16 | H2 |
| `c0_uniform` | A, B, C with `allocation=uniform`, CIFAR-10 90%, 3 seeds | 9 | H5 |
| `midckpt` | the CIFAR-10 90% \|w\| seed-0 run again (C0 had no mid-training checkpoints) | 1 | H4 (x-axis) |
| `dense_study` | dense reference per (dataset, model), 1 seed (a reference, not a tested arm) | 3 | ceiling |
| `regimes_tin` | key criteria + power seeds + dense on Tiny ImageNet 90% (optional) | 17 | H0–H4 at larger scale |

Suite `study` = `midckpt`, `c0_uniform`, `regimes`, `power`, `dense_study`, run seed-major (seed 0
of every cell first). On one RTX 3090 (≈ 24 min per ResNet-18 run, ≈ 15 min per VGG-16 run):
**92 new runs ≈ 34 h**, of which the seed-0 pass (28 runs) ≈ 10 h. `regimes_tin` ≈ 29 h more.

## Analyses and decision rules

- **Per regime, per criterion vs |w|:** difference of mean final-epoch test accuracy, 95% Welch CI,
  two-sided Welch test; **Holm** adjustment over all confirmatory comparisons of the table
  (exploratory rows excluded) → `paper/tables/regimes.tex`, `results/regimes_summary.json`.
  H0, H1: "worse" = Δ < 0 with p_Holm < 0.05.
- **Equivalence (H2):** TOST, α = 0.05, margins **±0.3 pp (CIFAR-10), ±0.5 pp (CIFAR-100, Tiny
  ImageNet)** (≈ 2× the usual seed-to-seed std at these accuracies). "Equivalent" only if TOST
  rejects; "no difference" from a non-significant Welch test is never reported as equivalence.
- **H4:** `analysis/score_overlap.py` on the epoch-50 and epoch-100 checkpoints replays the loop's
  next prune step exactly (its swap fraction, entropy from `entropy_batches` batches, Taylor/Fisher/|g|
  averaged over `update_interval` batches); divergence = 1 − Jaccard with |w|'s drop set, averaged over
  the two epochs; Spearman ρ with p over the confirmatory (criterion, regime) points
  → `paper/Figures/divergence_accuracy.pdf`, `results/divergence.json`.
- **H5:** B − A under global vs uniform allocation → `paper/tables/allocation.tex`.
- **E1 / what H measures:** per-channel share of exact zeros and Spearman(H, zero share), from the
  same script (`spearman_H_zero_pooled`).
- **FLOPs:** every table reports the sparse FLOPs reduction next to accuracy.
- The C0 screen (|Δ| > 1.96 × pooled std) is kept in the JSON for continuity with C0.
- No hyper-parameter is tuned on the test set.

## Run order

```bash
python -m experiments.plan run study         # seed-major; ~34 h on one RTX 3090
python -m experiments.plan run regimes_tin   # optional, ~29 h
python analysis/score_overlap.py results/checkpoints/*_ep*.pt results/midckpt/checkpoints/*_ep*.pt
python aggregate.py
```

The runner skips finished runs, so `study` can be stopped and restarted (the interrupted run restarts
from epoch 1).

## Changelog

- 2026-09-28: design written after C0 and the overlap diagnostics.
- 2026-09-28: single GPU; added suites `study` (one command, priority order) and `dense_study`
  (dense references of the regimes only). No change to hypotheses or analyses.
- 2026-09-28: `--parallel N` runs N jobs at once on the one GPU (spare VRAM). Each run's config is
  unchanged, so accuracy/sparsity/FLOPs are unaffected; epoch times and the speed protocol are not
  comparable to runs made alone. Any speed or cost number used in the paper is re-measured with
  `bench.py` on an idle GPU.
- 2026-09-28: measured on the RTX 3090: 3 parallel runs take ≈ 22.7 s/epoch each vs ≈ 7.5 s alone,
  i.e. no throughput gain; `study` is run one job at a time.
- 2026-09-28: `study` runs seed-major (seed 0 of every cell, then seeds 1 and 2); the seed-0 pass is
  used only to catch broken runs, not to change the design. Trainer speed-ups that do not change
  the method: no host sync per step (running sums on the device, topology-update time measured with
  CUDA events and read once per epoch) and `cudnn.benchmark`. Runs are not bitwise deterministic
  before or after this change; C0 runs made before it are reused as they are. Epoch times and
  `topology_update_seconds` of runs before and after the change are not comparable.
- 2026-09-28: **design revision, before any result of the study was looked at** (only three
  interrupted |w| runs had logged up to epoch 60; no study run had finished). Reasons: a review of
  the design found four gaps. (1) No lower anchor → added the random-drop control (`dst_random`,
  own RNG, H0). (2) "No difference" is not equivalence → Welch CIs, Holm, TOST with fixed margins,
  and 5 seeds for |w| and H·|w| (`power`). (3) One architecture → VGG-16 on CIFAR-100 90% as a
  secondary regime. (4) The H4 replay did not match the loop (fixed 20 % fraction, 4 batches for
  gradient statistics) → it now uses the loop's own fraction and averaging window, stored in the
  mid-training checkpoints. Added the exploratory E1 (robust entropy estimate), motivated by the C0
  diagnostics. To fit one GPU: the full criterion set only in the two main regimes, the key set in
  the secondary ones; dense references with 1 seed; mid-training checkpoints saved by the regular
  |w| seed-0 runs instead of separate runs; `criteria` (ELTP, RigL, movement) dropped from `study`
  (still runnable on its own); test set evaluated every 10 epochs (the reported accuracy is always
  the last epoch's, which is evaluated). Tiny ImageNet images are cached decoded (same pixels).
- 2026-09-30: **results.** All 100 runs of `study` + `c0` finished; the C0 configurations were
  re-run in full with the final code (the first C0, which motivated this design, gave C − A = −0.04 pp,
  B − A = −2.45 pp, A = 95.15 %; the re-run gives C − A = +0.27 pp, B − A = −2.59 pp, A = 95.03 %;
  neither passes the C0 rule). The paper reports the re-run and mentions the first run.
  `regimes_tin` and `criteria` were not run; the Tiny ImageNet regime is left out of the tables.
- 2026-09-30: **post hoc analyses** (added after seeing the results, reported as such): (a) H alone vs
  the random-drop control per regime (`entropy_vs_random.tex`), because H loses far more than random
  at the same divergence from |w|; (b) H4 robustness: the divergences form two clusters (≈0.3 for
  H·|w|, Taylor, Fisher; ≈0.93 for H, |g|, random), so the pooled Spearman mostly reflects near vs
  far; within-regime ρ and cluster ranges are stored in `divergence.json` (`post_hoc`).
  Note: with uniform allocation, B ends at 90.17–90.33 % measured sparsity (weights regrown where the
  gradient is exactly zero stay zero); stated in `allocation.tex`.
