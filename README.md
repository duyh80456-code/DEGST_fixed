# DEGST: Dynamic Entropy-Guided Sparse Training

Code for the revised DEGST paper (`paper/`). One training loop (`degst/trainer.py`) runs every
method, so methods differ only by the overrides listed in `degst/config.py::METHODS`.

### History: the pre-revision code

The first implementation was **not** what the paper describes. Its "DEGST" score was
`|w·g|·(1 − E/H_max)`, where E is an entropy over the layer's weight-magnitude distribution
rather than activation entropy, and it had no structured, hybrid, Slimming, FPGM, Movement or
AutoPruner code. Everything it could do is now a method of the unified framework:

| Old code | Now |
| --- | --- |
| `utils/dynamic_trainer.py` (first DEGST) | method `degst_v0`; its score alone in the current loop: `dst_eltp` |
| `analyze.py`: train 50 ep → one-shot prune → LTH rewind → fine-tune 30 ep | `posttrain_magnitude`, `posttrain_layerwise`, `posttrain_snip`, `posttrain_eltp` (suite `legacy`, table `paper/tables/legacy.tex`) |
| `quick_eval.py` | suite `quick` |
| `test.py`, `demo_eltp_quick.py` | `tests/test_degst.py` |
| `benchmark.py`, `utils/metrics.py` | `bench.py`, `degst/metrics.py` (old FLOPs counter tracked spatial size sequentially and was wrong for ResNet shortcuts) |
| `plot_results.py`, `config.py`, `main.py` | `aggregate.py`, `degst/config.py`, `train.py` |

The old sources are kept unchanged in `archive/legacy_code.zip`, and the paper as it was before the
revision in `archive/paper_before_revision.zip`.

## Layout

| Path | Contents |
| --- | --- |
| `degst/entropy.py` | per-channel histogram entropy H(A), collected with hooks during training |
| `degst/criteria.py` | drop criteria: magnitude, entropy, entropy·magnitude, Taylor, Fisher, movement, gradient, SNIP |
| `degst/sparse.py` | unstructured masks: global / uniform / ERK allocation, prune–regrow, gradual (cubic) pruning |
| `degst/structured.py` | channel gates on DepGraph groups + **physical** channel removal (Torch-Pruning) |
| `degst/metrics.py` | params, FLOPs (sparse and dense-executed), speed protocol |
| `train.py` | one run → `results/<run_id>.json` (+ checkpoint) |
| `experiments/plan.py` | all suites of the revision plan; resumable runner, sharding over GPUs |
| `aggregate.py` | mean ± std tables → `paper/tables/*.tex`, Pareto plot → `paper/Figures/`, C0 decision |
| `bench.py` | re-time saved checkpoints on another GPU |
| `analysis/score_overlap.py` | replays a prune step of the loop with every criterion: drop-set overlap, allocation between layers, what H measures (zero share) |
| `tests/test_degst.py` | unit tests; `--smoke` trains every method briefly on random data |
| `archive/` | pre-revision code and paper (zip, read-only reference) |

## Order of work (after C0)

C0 was negative: H(A)·|w| does not beat |w| (−0.04 pp), and H(A) alone loses 2.45 pp. The paper is
now a controlled study of drop criteria in DST. Hypotheses, suites and decision rules are fixed in
[`experiments/DESIGN.md`](experiments/DESIGN.md), written before the new runs.

```bash
pip install scipy                            # aggregate.py: Welch CIs, TOST, Holm, Spearman
python -m experiments.plan run study         # one GPU, 92 runs ~34 h, seed 0 of every cell first (~10 h)
python -m experiments.plan run regimes_tin   # Tiny ImageNet 90%: |w|, H, random x 3 seeds + dense, 100 epochs (10 runs, ~14 h)
python analysis/score_overlap.py results/checkpoints/*_ep*.pt results/midckpt/checkpoints/*_ep*.pt
python aggregate.py            # + paper/tables/regimes.tex, allocation.tex, Figures/divergence_accuracy.pdf
```

## Original order of work (before C0)

The plan is sized for one consumer GPU: 172 core runs (~34k epochs). There is no per-method
learning-rate tuning; every method uses lr 0.1. Main points (50/75/90%), C0 and the extension use
3 seeds; the extra Pareto points (60/80/95%) and sensitivity variations use 1 seed.

```bash
pip install -r requirements.txt   # needs a CUDA build of torch for GPU training

# 1. C0: entropy ablation (9 runs). Decides whether the rest is worth running.
python -m experiments.plan run c0
python aggregate.py            # prints "C0: ... => POSITIVE/NEGATIVE", writes results/c0_decision.json

# 2. Dense references + main table + Pareto sweep (120 runs; `pareto` includes `main`)
python -m experiments.plan run dense
python -m experiments.plan run pareto

# 3. Sensitivity (7 runs) and CIFAR-100 / MobileNetV2 at 90% (30 runs)
python -m experiments.plan run sensitivity
python -m experiments.plan run ext

python aggregate.py            # regenerate all tables/figures in paper/
```

`python -m experiments.plan status all` shows progress and the plan size. On several GPUs:
`python -m experiments.plan run pareto --shard 0/2 --device cuda:0` (and 1/2 on the other).
`list <suite>` prints plain `train.py` commands for a cluster scheduler or rented GPU.

If C0 is negative, run `criteria` (27 runs; route 1: a study of importance criteria in one
prune/regrow loop; `dst_taylor` and `dst_fisher` are also the perturbation-aware scores of route 2).
`legacy` (10 runs, optional) re-runs the pre-revision pipelines for comparison with the old numbers.

## Datasets

`--dataset cifar10 | cifar100 | tinyimagenet` (plus `fake` for smoke tests). All three are
downloaded to `--data-root` (default `./data`) on first use. Tiny ImageNet (200 classes, 64×64,
~240 MB from `cs231n.stanford.edu`) uses its official validation set as the test set; the
backbones are the same CIFAR-style networks, so a 64×64 input costs ~4× the FLOPs.

```bash
python train.py --method degst --dataset tinyimagenet --sparsity 0.9 --seed 0
```

## Definitions that the numbers depend on

- **Sparsity** = 1 − (non-zero conv/linear weights) / (conv/linear weights of the dense model).
  Structured and hybrid budgets use the same definition, so all methods at "90%" keep the same
  number of weights (structured: up to one channel; hybrid: exactly).
- **Hybrid**: the structured stage removes `hybrid_struct_share × target` of the weights (default
  half) by the midpoint of the update window; the channels are then removed physically and
  unstructured DEGST fills the rest of the budget inside the compact model.
- **FLOPs**: `flops_sparse` counts non-zero weights only (idealised sparse kernels);
  `flops_dense` counts what dense kernels execute (only channel removal reduces it). Global
  allocation prunes late, low-resolution layers hardest, so its FLOPs reduction is below its
  parameter sparsity; uniform allocation gives FLOPs reduction ≈ sparsity.
- **Accuracy** is the final-epoch test accuracy, never the maximum over epochs.
- **Speed**: fp32, eval mode, synthetic input on device, no data loading, cudnn.benchmark, 50
  warm-up + 200 timed iterations with CUDA events; median latency per forward pass at batch
  1/128/1024, throughput = batch / median latency (same measurement). Unstructured masks run on
  dense kernels, so their latency is that of the dense model.
- **Training cost**: wall-clock per epoch, GPU-hours, and the share of time spent in topology
  updates and in entropy histograms are stored under `cost` in every result file.

## Tests

```bash
python tests/test_degst.py            # unit tests, ~30 s on CPU, no dataset needed
python tests/test_degst.py --smoke    # + every method for 3 epochs on random data (~15 min on CPU)
python -m experiments.plan run quick  # 5 epochs on real CIFAR-10 for dense/degst/rigl/degst_s
```

## Hướng dẫn nhanh (tiếng Việt)

**Sau C0 (âm tính):** bài chuyển thành nghiên cứu các tiêu chí cắt trong DST. Thiết kế, giả thuyết và
quy tắc quyết định nằm trong `experiments/DESIGN.md`. Trên 1 GPU chỉ cần `python -m experiments.plan run study`
(92 run, khoảng 34 giờ trên RTX 3090, dừng rồi chạy lại được), tùy chọn thêm `regimes_tin`
(Tiny ImageNet thu gọn: |w|, H, random × 3 seed + dense, 100 epoch = cùng số iteration với CIFAR, 10 run, khoảng 14 giờ). `study` chạy seed 0 của mọi ô trước (khoảng 10 giờ) để phát hiện lỗi sớm.
Cần `pip install scipy` cho phần thống kê (Welch, TOST, Holm).
`--parallel N` vẫn có nhưng trên RTX 3090 không nhanh hơn (đã đo: 3 run song song ≈ 22.7 s/epoch mỗi run). Sau đó chạy `analysis/score_overlap.py` trên các checkpoint `_ep50`/`_ep100`
và `python aggregate.py`.

**Kế hoạch ban đầu (trước C0):**

1. Cài đặt: `pip install -r requirements.txt`, kiểm tra bằng `python tests/test_degst.py`.
2. Chạy **C0 trước**: `python -m experiments.plan run c0`, rồi `python aggregate.py`. Dòng `C0: ...`
   cho biết entropy có giúp hơn `|w|` hay không (quy tắc `C − A > 1.96 × std gộp`).
3. Nếu dương tính: `dense` → `pareto` → `sensitivity` → `ext` (xem mục *Order of work*).
   Nếu âm tính: chạy `criteria` và chuyển bài sang nghiên cứu so sánh tiêu chí.
   Kế hoạch đã rút gọn cho 1 GPU phổ thông (172 lần train cốt lõi), không tune LR, điểm phụ chạy 1 seed.
4. Muốn đối chiếu với số liệu cũ: `python -m experiments.plan run legacy`.
5. Sau mỗi đợt chạy, `python aggregate.py` sinh lại toàn bộ bảng trong `paper/tables/` và hình Pareto;
   không sửa số trong bài bằng tay. Các chỗ `\TODO{...}` (chữ đỏ) trong bài phải được viết lại theo kết quả.
6. Chạy nhiều GPU: thêm `--shard i/n --device cuda:i`; xem tiến độ bằng `python -m experiments.plan status <suite>`.
