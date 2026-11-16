"""y = x @ w.T + b with BF16 inputs/outputs and FP32 accumulation (the precision the rules prescribe)."""
import torch


def reference(x, w, b):
    return (x.float() @ w.float().T + b.float()).to(torch.bfloat16)


def exact(x, w, b):
    return x.double() @ w.double().T + b.double()
