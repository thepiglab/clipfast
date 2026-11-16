"""Training-step benchmark for BiomedCLIP contrastive fine-tuning (fwd + bwd + AdamW step), one GPU.

Impls:
  oc_fp32   open_clip model, fp32, torch AdamW (foreach), text padded to 256  (naive)
  oc_amp    open_clip model, autocast bf16 (open_clip --precision amp_bf16), AdamW foreach, text padded to 256
  ft_trim   oc_amp + text trimmed to the batch's longest caption (bucketed) + fused AdamW
  ft_comp   ft_trim + torch.compile(model) (inductor fuses LN/GELU/bias/residual; SDPA attention)
Also checks: per-impl gradient cosine vs fp32 on one step, and loss after a few steps on a fixed batch.
"""
import argparse
import copy
import json
import os
import socket
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clipfast.load import load_reference  # noqa: E402
from bench import text_ids  # noqa: E402
from clipfast.train_model import LeanBiomedCLIP  # noqa: E402

LEN_BUCKETS = [16, 32, 48, 64, 96, 128, 192, 256]


def clip_loss(fi, ft, scale):
    fi, ft = F.normalize(fi.float(), dim=-1), F.normalize(ft.float(), dim=-1)
    logits = scale.exp() * fi @ ft.T
    lab = torch.arange(fi.shape[0], device=fi.device)
    return (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2


def trim(ids, pad=0):
    L = int((ids != pad).sum(1).max())
    Lb = next(b for b in LEN_BUCKETS if L <= b)
    return ids[:, :Lb].contiguous()


def make_step(model, impl):
    fused = impl in ("ft_trim", "ft_comp") or impl.startswith("lean")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.1,
                            fused=fused, foreach=not fused)
    amp = impl != "oc_fp32"
    fwd = model
    if impl in ("ft_comp", "lean_comp"):
        fwd = torch.compile(model, dynamic=False)

    def step(x, ids):
        if impl in ("ft_trim", "ft_comp", "lean_pad", "lean_comp"):
            ids = trim(ids)
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            if impl == "lean_pack":
                pk = LeanBiomedCLIP.pack(ids)
                pk = [t.to(x.device, non_blocking=True) for t in pk[:5]] + [pk[5]]
                fi, ft, scale = fwd(x, packed=pk)
            else:
                fi, ft, scale = fwd(x, ids)[:3]
        loss = clip_loss(fi, ft, model.logit_scale)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss
    return step, opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--impls", default="oc_fp32,oc_amp,ft_trim,ft_comp,lean_pad,lean_pack,lean_comp")
    ap.add_argument("--batches", default="128,256")
    ap.add_argument("--lens", default="77,pmc,256")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meta = dict(host=socket.gethostname(), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                job=os.environ.get("SLURM_JOB_ID"))
    out = open(a.out, "a")

    def rec(**kw):
        kw.update(meta)
        out.write(json.dumps(kw) + "\n")
        out.flush()
        print(json.dumps(kw), flush=True)

    base, _, _ = load_reference()
    base.train()
    for m in base.modules():          # deterministic comparison: no dropout
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    vocab = base.text.config.vocab_size
    gen = torch.Generator().manual_seed(0)
    dev = torch.device("cuda")

    # ---- gradient check (one step, B=32, L=64 mixed) ----
    x = torch.randn(32, 3, 224, 224, device=dev)
    ids = text_ids(32, 64, vocab, True, gen).to(dev)
    grads = {}
    for impl in ["oc_fp32", "oc_amp", "ft_trim"]:
        m = copy.deepcopy(base)
        idt = trim(ids) if impl == "ft_trim" else ids
        with torch.autocast("cuda", torch.bfloat16, enabled=impl != "oc_fp32"):
            fi, ft, _ = m(x, idt)[:3]
        clip_loss(fi, ft, m.logit_scale).backward()
        grads[impl] = torch.cat([p.grad.flatten().float() for p in m.parameters() if p.grad is not None])
        del m
    for impl in ["oc_amp", "ft_trim"]:
        rec(kind="train_gradcheck", impl=impl,
            grad_cos=float(F.cosine_similarity(grads[impl], grads["oc_fp32"], dim=0)))
    del grads
    lean = LeanBiomedCLIP(base).cuda()
    with torch.no_grad():
        fi0, ft0, _ = base(x, ids)[:3]
        fi1, ft1, _ = lean(x, trim(ids))
        pk = LeanBiomedCLIP.pack(ids)
        pk = [t.cuda() for t in pk[:5]] + [pk[5]]
        with torch.autocast("cuda", torch.bfloat16):
            ft2 = lean.encode_text_packed(*pk).float()
            fi3, ft3, _ = lean(x, trim(ids))
    cs = lambda a, b: float(F.cosine_similarity(a.float(), b.float(), dim=-1).min())
    rec(kind="train_parity", lean_fp32_img_cos=cs(fi0, fi1), lean_fp32_txt_cos=cs(ft0, ft1),
        lean_packed_bf16_txt_cos=cs(ft0, ft2), lean_bf16_img_cos=cs(fi0, fi3), lean_bf16_txt_cos=cs(ft0, ft3))
    with torch.autocast("cuda", torch.bfloat16):
        fi, ft, _ = lean(x, packed=pk)
    clip_loss(fi, ft, lean.logit_scale).backward()
    g = torch.cat([q.grad.flatten().float() for q in lean.parameters() if q.grad is not None])
    rec(kind="train_parity", lean_packed_grad_finite=bool(torch.isfinite(g).all()), lean_grad_norm=float(g.norm()))
    del lean
    torch.cuda.empty_cache()

    # ---- timing ----
    for B in [int(b) for b in a.batches.split(",")]:
        for L in [int(l) if l.isdigit() else l for l in a.lens.split(",")]:
            x = torch.randn(B, 3, 224, 224, device=dev)
            ids = text_ids(B, L, vocab, L != "pmc", gen).to(dev)
            for impl in a.impls.split(","):
                m = LeanBiomedCLIP(base).cuda() if impl.startswith("lean") else copy.deepcopy(base)
                torch.cuda.reset_peak_memory_stats()
                try:
                    step, _ = make_step(m, impl)
                    t0 = time.perf_counter()
                    for _ in range(3):
                        loss = step(x, ids)
                    torch.cuda.synchronize()
                    warm = time.perf_counter() - t0
                    ts = []
                    for _ in range(3):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        for _ in range(5):
                            loss = step(x, ids)
                        torch.cuda.synchronize()
                        ts.append((time.perf_counter() - t0) / 5 * 1e3)
                    ms = statistics.median(ts)
                    rec(kind="train", impl=impl, B=B, L=L, ms=ms, samples_per_s=B / ms * 1e3,
                        peak_gb=torch.cuda.max_memory_allocated() / 1e9, warm_s=warm, loss=float(loss))
                except torch.OutOfMemoryError:
                    rec(kind="train", impl=impl, B=B, L=L, oom=True)
                del m
                torch._dynamo.reset()
                torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
