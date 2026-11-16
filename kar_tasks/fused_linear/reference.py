"""Reference at the prescribed precision: bf16 inputs, fp32 accumulate + fp32 epilogue, one bf16 rounding at the end.
(The baseline rounds after the GEMM and again after the epilogue, as PyTorch does with separate kernels.)"""
import torch


def _epi(acc, b, r, mode):
    y = acc + b
    if mode == 1:
        y = torch.nn.functional.gelu(y)
    elif mode == 2:
        y = y + r
    return y


def reference(x, w, b, r, mode):
    return _epi(x.float() @ w.float().T, b.float(), r.float(), mode).to(torch.bfloat16)


def exact(x, w, b, r, mode):
    return _epi(x.double() @ w.double().T, b.double(), r.double(), mode)
