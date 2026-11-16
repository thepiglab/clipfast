"""Baseline: what PyTorch eager runs: cuBLAS linear (bias fused by cuBLASLt), then GELU or a residual add kernel."""
import torch.nn.functional as F


def kernel_fn(x, w, b, r, mode):
    y = F.linear(x, w, b)
    if mode == 1:
        return F.gelu(y)
    if mode == 2:
        return r + y
    return y
