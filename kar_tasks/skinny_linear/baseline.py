"""Baseline: cuBLAS through F.linear (what PyTorch / open_clip run)."""
import torch.nn.functional as F


def kernel_fn(x, w, b):
    return F.linear(x, w, b)
