"""Triton GEMM with fused epilogue (bias / bias+exact GELU / bias+residual), one BF16 rounding, FP32 accumulate.

Grouped (L2-friendly) tile order; per-shape-class fixed configs chosen at import (no autotune at call time).
"""
import torch
import triton
import triton.language as tl

CONFIGS = {   # (BM, BN, BK, GROUP_M, warps, stages) per N-class
    "wide": (256, 128, 32, 8, 8, 3),
    "narrow": (128, 64, 64, 8, 4, 3),
    "small": (64, 64, 64, 8, 4, 3),
}


@triton.jit
def _fused_linear(x_ptr, w_ptr, b_ptr, r_ptr, y_ptr, M, N, K,
                  sxm, sxk, swn, swk, srm, srn, sym, syn,
                  MODE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    per_group = GROUP_M * num_n
    gid = pid // per_group
    first_m = gid * GROUP_M
    gsize = min(num_m - first_m, GROUP_M)
    pm = first_m + (pid % per_group) % gsize
    pn = (pid % per_group) // gsize
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    xp = x_ptr + rm[:, None] * sxm + rk[None, :] * sxk
    wp = w_ptr + rn[None, :] * swn + rk[:, None] * swk
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        kmask = rk < K - k * BK
        a = tl.load(xp, mask=(rm[:, None] < M) & kmask[None, :], other=0.0)
        b = tl.load(wp, mask=(rn[None, :] < N) & kmask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        xp += BK * sxk
        wp += BK * swk
    bias = tl.load(b_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    if MODE == 1:
        acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))
    omask = (rm[:, None] < M) & (rn[None, :] < N)
    if MODE == 2:
        res = tl.load(r_ptr + rm[:, None] * srm + rn[None, :] * srn, mask=omask, other=0.0).to(tl.float32)
        acc = acc + res
    tl.store(y_ptr + rm[:, None] * sym + rn[None, :] * syn, acc.to(tl.bfloat16), mask=omask)


def _cfg(N):
    return CONFIGS["wide"] if N >= 2048 else CONFIGS["narrow"]


def kernel_fn(x, w, b, r, mode):
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    BM, BN, BK, G, nw, ns = CONFIGS["small"] if M < 2048 else _cfg(N)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    rr = r if mode == 2 else y
    _fused_linear[grid](x, w, b, rr, y, M, N, K, x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                        rr.stride(0), rr.stride(1), y.stride(0), y.stride(1),
                        MODE=mode, BM=BM, BN=BN, BK=BK, GROUP_M=G, num_warps=nw, num_stages=ns)
    return y


def _warm():          # compile every specialisation at import, never per call
    dev = torch.device("cuda")
    for mode in (0, 1, 2):
        for M, N, K in ((4096, 3072, 768), (4096, 768, 3072), (4096, 768, 768), (256, 2304, 768)):
            x = torch.zeros(M, K, device=dev, dtype=torch.bfloat16)
            w = torch.zeros(N, K, device=dev, dtype=torch.bfloat16)
            b = torch.zeros(N, device=dev, dtype=torch.bfloat16)
            r = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
            kernel_fn(x, w, b, r, mode)
    torch.cuda.synchronize()


_warm()
