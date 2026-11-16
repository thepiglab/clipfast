"""PyTorch CPU baseline: open_clip BiomedCLIP fp32 (and bf16 autocast) on N threads. python cpu_torch_bench.py N OUT"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clipfast.load import load_reference  # noqa: E402

n, out = int(sys.argv[1]), sys.argv[2]
torch.set_num_threads(n)
ref, _, tok = load_reference("cpu")
f = open(out, "a")
for amp in (False, True):
    for B in (1, 16):
        x = torch.randn(B, 3, 224, 224)
        with torch.inference_mode(), torch.autocast("cpu", torch.bfloat16, enabled=amp):
            ref.encode_image(x)
            t0 = time.perf_counter()
            reps = max(2, 16 // B)
            for _ in range(reps):
                ref.encode_image(x)
            ms = (time.perf_counter() - t0) / reps * 1e3
        r = dict(kind="image", impl=f"torch_cpu_{'bf16amp' if amp else 'fp32'}_t{n}", B=B, ms=ms, img_per_s=B / ms * 1e3)
        f.write(json.dumps(r) + "\n"); print(json.dumps(r), flush=True)
    ids = tok(["chest x-ray showing bilateral opacities and small effusions in the lower lobes"] * 64)
    with torch.inference_mode(), torch.autocast("cpu", torch.bfloat16, enabled=amp):
        ref.encode_text(ids)
        t0 = time.perf_counter()
        ref.encode_text(ids)
        ms = (time.perf_counter() - t0) * 1e3
    r = dict(kind="text", impl=f"torch_cpu_{'bf16amp' if amp else 'fp32'}_t{n}", B=64, L="pad256", ms=ms, txt_per_s=64 / ms * 1e3)
    f.write(json.dumps(r) + "\n"); print(json.dumps(r), flush=True)
