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

## Phase 2: activation information through energy instead of entropy

Written 2026-10-26, after the study's results and **before any phase-2 run**. Phase 2 is a separate,
follow-up experiment; its results are reported apart from the confirmatory study above.

**Question.** The study shows that entropy is the wrong way to use activations (it mostly measures
post-ReLU sparsity). Does activation information help once it is used through its energy, and does a
score that stays anchored to magnitude beat |w|?

**Step 0 (done before writing this section).**
- The existing `fisher` criterion is already the OBD form ½·mean_t(g²)·w² (`degst/criteria.py`), so the
  proposed "S2 = F̂·w²" is not a new configuration; it is run again in the screen only as a known
  comparison. (Its F̂ uses mini-batch rather than per-sample gradients.)
- Gap Dense − |w| (dense: 1 seed): CIFAR-100 ResNet-18 90 % +1.95 pp, Tiny ImageNet 90 % +1.82,
  CIFAR-10 98 % +1.56, CIFAR-10 90 % +0.29, VGG-16 CIFAR-100 −0.01. Regimes with |w| at the ceiling
  cannot show a gain, so the screen uses the two largest gaps: CIFAR-100 ResNet-18 and Tiny ImageNet.

**Scores.**
- S1 `dst_energy`: |w|·√E[a²] per input channel. This is the score of Wanda (Sun et al., 2024) applied
  in a DST loop with global ranking; it is a baseline, not a contribution.
- S2 `dst_fisher`: the existing Fisher criterion (see above).
- S3 `dst_composite`: |w|·(1 + β·ũ), ũ = √E[a²] min-max normalised to [0, 1] within each layer;
  β = 0 is |w|. Default β = 0.5.

**Where.** All selection (screen and β sweep) uses a validation split held out from the training set
(5k images on CIFAR-100, 10k on Tiny ImageNet; `--val-size`), never the test set. Same loop, recipe,
seeds 0–2 and (on Tiny ImageNet) the 100-epoch iteration-matched recipe as the study.

**Hypotheses.**
- H6: S1 beats |w| by the screen rule below.
- H7: S3 with β = 0.5 beats |w| by the screen rule below.
- H8: over β ∈ {0.1, 0.25, 0.5, 1, 2}, validation accuracy is highest at some β < 2 whose difference to
  |w| has a 95 % CI above zero (a peak at small, non-zero divergence from |w|).

**Screen rule (fixed now).** A score wins if, on the validation split, Δ = acc − acc(|w|) > 0 with a
95 % Welch CI (3 seeds) excluding zero in at least one screen regime, and it is not significantly worse
(CI below zero) in the other. This is a screen without multiplicity correction; it decides only what
goes into the confirmatory step.

**Stop rule.** If no score wins the screen, phase 2 stops: no further scores or β values are tried, and
the paper reports the negative screen ("even through energy, activation information does not beat |w|
in this DST loop"). The β sweep runs only if S3 wins or is within its CI of winning in both regimes
(Δ > 0 in both); β is then chosen as the validation argmax and fixed before the confirmatory step.

**Confirmatory step.** The winning score(s) (at most two, with the β chosen on validation) are run on
the test set in all five regimes, 3 seeds (`CONFIRM` in `experiments/plan.py`, suite `confirm`), and
analysed with the study's framework: Welch CIs, Holm over the new comparisons, TOST with the same
margins (±0.3 pp on CIFAR-10, ±0.5 pp on CIFAR-100 and Tiny ImageNet).

**Corrections to the proposal this section is based on.** (1) S2 already exists (above). (2) The
equivalence margins are ±0.3 / ±0.5 pp; 0.31 pp was the C0 threshold, not a margin. (3) The proposal
screened on the test set and then confirmed on the same test set; here the screen and the β choice use
a validation split, so the test set enters only the confirmatory step.

**Cost (RTX 3090, estimates).** Screen: 24 runs, ≈ 18 h (CIFAR-100 ≈ 4 h, Tiny ImageNet ≈ 14 h).
β sweep: 24 new runs, ≈ 18 h. Confirmatory: ≈ 9 h per winning score.

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
- 2026-09-30: **Tiny ImageNet, reduced** (decided before any Tiny ImageNet run; motivated only by GPU
  budget). `regimes_tin` = |w|, H, random × seeds 0–2 on ResNet-18 at 90 %, + dense (1 seed): 10 runs,
  seed-major. The |w| seed-0 run saves mid-training checkpoints (epochs 50, 100) for the overlap
  replay. It tests H0, H1 and the (post hoc) H-vs-random comparison at a larger scale; H·|w| and the
  extra seeds are not run, so H2 is not tested on Tiny ImageNet. All 10 runs are completed and reported
  whatever their outcome; the two comparisons join the Holm family of `regimes.tex` (as the design
  foresaw for this regime), so the adjusted p-values of the other regimes will change slightly.
- 2026-09-30: **Tiny ImageNet recipe matched in iterations** (decided after one epoch of the dense
  run had shown 48.8 s/epoch, before any sparse Tiny ImageNet result; that dense run was stopped and
  is not used). Tiny ImageNet has 781 iterations per epoch against 390 on CIFAR, so all epoch counts
  are halved: 100 epochs, warm-up 5, T_end 75, mid-training checkpoints at epochs 25 and 50 (`TIN_RECIPE`
  in `experiments/plan.py`); the dense reference also trains 100 epochs. In iterations the loop is the
  same as on CIFAR: ~78k SGD steps, warm-up ~3.9k, T_end ~58.6k, ~136 topology updates at ΔT = 400,
  update fraction 16.2 % / 5.7 % at the two checkpoints. Batch size, learning rate and every other
  setting are unchanged. Cost ≈ 1.4 h per run, ≈ 14 h for the 10 runs.
- 2026-10-01: **Tiny ImageNet results.** All 10 runs finished (137 topology updates each, sparsity
  exactly 90 %). |w| 63.80 ± 0.13, H 55.00 (−8.80 pp, p_Holm 0.003), random 61.47 (−2.33 pp, CI
  [−3.63, −1.02], p_Holm 0.15), dense 65.62 (1 seed); H − random −6.47 pp (p_Holm 0.001). With the two
  new comparisons in the Holm family the adjusted p-values of the CIFAR regimes moved slightly (e.g. H on
  CIFAR-10 90 %: 0.041 → 0.045); no conclusion changed. H4: ρ = −0.63 over 20 points (p = 0.003).
  Bug fixed: train.py parsed `--end-epoch 75` as 75.0 while the plan kept 75, so the run ids differed
  (`c8231824` vs `aa812d96`) and the tables could not find the runs; `make_config` now casts ints given
  for float fields, and a test checks that plan and train.py agree on every run id. No run was retrained.
- 2026-10-01: **post hoc figures**: learning curves (train accuracy every epoch, test every 10 epochs;
  `learning_curves.pdf`, `results/learning_curves.json`), which show that H alone underfits in the four
  harder regimes, and the full Jaccard matrix of the main regimes (`jaccard_matrix.pdf`); plus a TikZ
  overview of the design in the paper. No new runs.
