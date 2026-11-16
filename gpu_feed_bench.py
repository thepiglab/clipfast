"""GPU feeding benchmark: nvJPEG batch decode (torchvision.io.decode_jpeg on cuda) + GPU resize/crop/normalize, vs
the PIL CPU path, and the pre-decoded uint8 cache path. Also parity of each path's pixels/embeddings vs open_clip's
PIL preprocess.  python gpu_feed_bench.py --out results/gpu_feed.jsonl
"""
import argparse
import io
import json
import os
import socket
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_feed_bench import make_jpegs  # noqa: E402
from clipfast.load import load_preprocess_only, load_reference  # noqa: E402
from clipfast.fast import FastBiomedCLIP  # noqa: E402

MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1)
STD = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1)


def gpu_prep(imgs_u8, size=224):
    """list of uint8 CHW cuda tensors -> normalized [B,3,224,224] (resize shortest side bicubic+antialias, crop)."""
    out = torch.empty((len(imgs_u8), 3, size, size), device="cuda")
    m, s = MEAN.cuda(), STD.cuda()
    for i, x in enumerate(imgs_u8):
        h, w = x.shape[-2:]
        sc = size / min(h, w)
        nh, nw = max(size, round(h * sc)), max(size, round(w * sc))
        y = F.interpolate(x[None].float(), size=(nh, nw), mode="bicubic", antialias=True)
        t, l = (nh - size) // 2, (nw - size) // 2
        out[i] = ((y[0, :, t:t + size, l:l + size].clamp(0, 255) / 255.0) - m[0]) / s[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torchvision.io as tio
    meta = dict(host=socket.gethostname(), gpu=torch.cuda.get_device_name(), job=os.environ.get("SLURM_JOB_ID"))
    out = open(a.out, "a")

    def rec(**kw):
        kw.update(meta)
        out.write(json.dumps(kw) + "\n")
        out.flush()
        print(json.dumps(kw), flush=True)

    jp = make_jpegs()
    pre = load_preprocess_only()
    ref, _, tok = load_reference()
    fast = FastBiomedCLIP(ref, tok, graphs=True)
    from PIL import Image
    for size, blobs in jp.items():
        datas = [torch.frombuffer(bytearray(b), dtype=torch.uint8) for b in blobs]
        # nvJPEG batched decode
        for _ in range(2):
            imgs = tio.decode_jpeg(datas, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = 5
        for _ in range(reps):
            imgs = tio.decode_jpeg(datas, device="cuda")
        torch.cuda.synchronize()
        dec = len(datas) * reps / (time.perf_counter() - t0)
        t0 = time.perf_counter()
        for _ in range(reps):
            px = gpu_prep(imgs)
        torch.cuda.synchronize()
        prep = len(datas) * reps / (time.perf_counter() - t0)
        t0 = time.perf_counter()
        for _ in range(reps):
            px = gpu_prep(tio.decode_jpeg(datas, device="cuda"))
            e = fast.encode_image(px)
        torch.cuda.synchronize()
        e2e = len(datas) * reps / (time.perf_counter() - t0)
        # parity vs PIL path
        ref_px = torch.stack([pre(Image.open(io.BytesIO(b)).convert("RGB")) for b in blobs[:16]]).cuda()
        with torch.inference_mode():
            e_ref = ref.encode_image(ref_px, normalize=True).float()
            e_gpu = fast.encode_image(gpu_prep(tio.decode_jpeg(datas[:16], device="cuda"))).float()
            e_pilfast = fast.encode_image(ref_px).float()
        rec(kind="gpu_feed", size=size, nvjpeg_decode_img_per_s=dec, gpu_prep_img_per_s=prep,
            decode_prep_encode_img_per_s=e2e,
            emb_cos_min_gpu_path_vs_openclip=float((e_gpu * e_ref).sum(-1).min()),
            emb_cos_min_pil_path_fast_vs_openclip=float((e_pilfast * e_ref).sum(-1).min()))
    # pre-decoded uint8 cache path: 224x224 uint8 already in pinned host memory -> GPU normalize -> encode
    B = 256
    cache = torch.randint(0, 255, (B, 3, 224, 224), dtype=torch.uint8).pin_memory()
    m, s = MEAN.cuda(), STD.cuda()
    for _ in range(3):
        fast.encode_image(((cache.cuda(non_blocking=True).float() / 255) - m) / s)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        fast.encode_image(((cache.cuda(non_blocking=True).float() / 255) - m) / s)
    torch.cuda.synchronize()
    rec(kind="cache_feed", B=B, img_per_s=B * 10 / (time.perf_counter() - t0),
        cache_bytes_per_img=3 * 224 * 224)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
