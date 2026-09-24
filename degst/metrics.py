"""Parameter, FLOPs and speed measurement.

Two FLOPs figures are reported for every model, because they answer different
questions:

* ``flops_dense``  -- FLOPs of the (possibly channel-pruned) architecture executed
  with dense kernels. This is what a GPU actually runs; unstructured zeros do
  not reduce it.
* ``flops_sparse`` -- FLOPs counting only non-zero weights (2 x MACs of non-zero
  weights x output positions). This is the idealised figure used in the sparse
  training literature (e.g. RigL) and the x-axis of the Pareto plots.

Both count conv and linear layers only (BN/ReLU/pooling are excluded, as is
standard). Because early layers have large spatial maps, removing a weight
there saves more FLOPs than removing one in a late layer: FLOPs reduction and
parameter sparsity coincide only for uniform per-layer sparsity.

Speed protocol (C3): eval mode, fp32, synthetic input already on the device, no
data loading, ``cudnn.benchmark=True``, ``warmup`` untimed iterations, then
``iters`` iterations timed individually with CUDA events (synchronised). For
each batch size b we report the median latency per forward pass and
throughput = b / median latency, i.e. both numbers come from the *same*
measurement and are consistent by construction (fixes C3.1).
"""
import statistics
import time
from typing import Dict, Iterable

import torch
import torch.nn as nn


def param_counts(model: nn.Module) -> Dict[str, int]:
    weights = [m.weight for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    return {
        'params_total': sum(p.numel() for p in model.parameters()),
        'weights_total': sum(w.numel() for w in weights),
        'weights_nonzero': int(sum((w.detach() != 0).sum().item() for w in weights)),
    }


@torch.no_grad()
def flops(model: nn.Module, input_size=(1, 3, 32, 32)) -> Dict[str, float]:
    dense, sparse, hooks = [0], [0], []

    def hook(m, _inp, out):
        w = m.weight.detach()
        positions = out.shape[2] * out.shape[3] if isinstance(m, nn.Conv2d) else 1
        dense[0] += 2 * positions * w.numel()
        sparse[0] += 2 * positions * int((w != 0).sum().item())

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    model(torch.zeros(input_size, device=dev))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return {'flops_dense': dense[0], 'flops_sparse': sparse[0]}


@torch.no_grad()
def measure_speed(model: nn.Module, batch_sizes: Iterable[int] = (1, 128, 1024), device='cuda',
                  warmup=50, iters=200, input_hw=32) -> Dict[str, Dict[str, float]]:
    model = model.to(device).eval()
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.backends.cudnn.benchmark = True
    out = {}
    for b in batch_sizes:
        x = torch.randn(b, 3, input_hw, input_hw, device=device)
        try:
            for _ in range(warmup):
                model(x)
            times = []
            if cuda:
                torch.cuda.synchronize()
                for _ in range(iters):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    model(x)
                    e.record()
                    torch.cuda.synchronize()
                    times.append(s.elapsed_time(e))
            else:
                for _ in range(iters):
                    t0 = time.perf_counter()
                    model(x)
                    times.append((time.perf_counter() - t0) * 1000)
        except RuntimeError as err:       # e.g. OOM at large batch on small GPUs
            out[str(b)] = {'error': str(err)[:200]}
            if cuda:
                torch.cuda.empty_cache()
            continue
        med = statistics.median(times)
        out[str(b)] = {
            'latency_ms_median': med,
            'latency_ms_mean': statistics.fmean(times),
            'latency_ms_std': statistics.pstdev(times),
            'throughput_img_s': b / med * 1000.0,
        }
    return out


def protocol(device, warmup, iters, batch_sizes) -> Dict:
    name = torch.cuda.get_device_name(0) if torch.device(device).type == 'cuda' else 'cpu'
    return {'device': name, 'dtype': 'fp32', 'warmup': warmup, 'iters': iters,
            'batch_sizes': list(batch_sizes), 'timer': 'cuda_events' if 'cpu' != name else 'perf_counter',
            'includes_data_loading': False, 'cudnn_benchmark': True, 'statistic': 'median',
            'torch': torch.__version__}
