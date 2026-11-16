"""Per-kernel time breakdown of the clipfast eager path (torch.profiler), to pick kar targets."""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clipfast.load import load_reference  # noqa: E402
from clipfast.fast import FastBiomedCLIP  # noqa: E402
from bench import text_ids  # noqa: E402


def classify(name):
    n = name.lower()
    for key, tag in [("gemm", "gemm"), ("cutlass", "gemm"), ("nvjet", "gemm"), ("sm90", "gemm"), ("sm100", "gemm"),
                     ("sm120", "gemm"), ("flash", "attention"), ("fmha", "attention"), ("cudnn", "attention"),
                     ("attention", "attention"), ("layer_norm", "layernorm"), ("layernorm", "layernorm"),
                     ("gelu", "gelu"), ("conv", "conv"), ("cat", "copy/cat"), ("copy", "copy/cat"),
                     ("elementwise", "elementwise"), ("index", "embed/index"), ("reduce", "reduce")]:
        if key in n:
            return tag
    return "other"


def prof(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    per = defaultdict(float)
    cls = defaultdict(float)
    for e in p.key_averages():
        t = e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
        if t <= 0:
            continue
        per[e.key] += t / iters
        cls[classify(e.key)] += t / iters
    tot = sum(cls.values())
    return tot, dict(sorted(cls.items(), key=lambda kv: -kv[1])), sorted(per.items(), key=lambda kv: -kv[1])[:15]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref, _, tok = load_reference()
    f = FastBiomedCLIP(ref, tok, graphs=False)
    vocab = ref.text.config.vocab_size
    g = torch.Generator().manual_seed(0)
    out = open(a.out, "a")
    cases = []
    for B in [1, 64, 512]:
        x = torch.randn(B, 3, 224, 224, device="cuda")
        cases.append((f"image_B{B}", lambda x=x: f.encode_image(x)))
    for B, L in [(1, 16), (64, 32), (512, 64), (256, 256)]:
        ids = text_ids(B, L, vocab, False, g).cuda()
        cases.append((f"text_B{B}_L{L}", lambda ids=ids: f.encode_text_ids(ids)))
    with torch.inference_mode():
        for name, fn in cases:
            tot, cls, top = prof(fn)
            r = dict(case=name, gpu_us=tot, by_class={k: round(v, 1) for k, v in cls.items()},
                     top=[(k[:110], round(v, 1)) for k, v in top])
            out.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
