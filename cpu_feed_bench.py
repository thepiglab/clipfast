"""CPU feeding benchmark (no GPU): can ~16 Zen 5 cores per GPU decode + preprocess images and tokenize captions as
fast as one RTX PRO 6000 consumes them (~7-8k img/s)?

Measures, per worker count (1, 4, 8, 16, 24 processes, pinned to one NUMA node's cores):
  - open_clip preprocess from JPEG bytes (PIL decode + RGB + bicubic resize + center crop + to-tensor + normalize)
  - the same with PIL draft mode (JPEG DCT-domain downscale while decoding; changes pixels slightly -> parity noted)
  - torchvision.io.decode_jpeg (libjpeg-turbo) + torch resize (antialias bicubic)
  - HF fast tokenizer throughput (batch encode, padding longest), single process and with TOKENIZERS_PARALLELISM
python cpu_feed_bench.py --out results/cpu_feed.jsonl
"""
import argparse
import io
import json
import multiprocessing as mp
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIZES = [(512, 512), (1024, 768), (2048, 1536), (3000, 2400)]
N_PER_SIZE = 64


def make_jpegs():
    from PIL import Image
    rng = np.random.default_rng(0)
    out = {}
    for (w, h) in SIZES:
        bufs = []
        for i in range(N_PER_SIZE):
            base = rng.random((h // 32 + 2, w // 32 + 2, 3))
            im = Image.fromarray((base * 255).astype("uint8")).resize((w, h), Image.BICUBIC)
            arr = np.asarray(im).astype(np.int16) + rng.integers(-12, 12, (h, w, 3))   # texture
            im = Image.fromarray(arr.clip(0, 255).astype("uint8"))
            b = io.BytesIO()
            im.save(b, format="JPEG", quality=90)
            bufs.append(b.getvalue())
        out[f"{w}x{h}"] = bufs
    return out


_PRE = None


def _init(kind, cpus):
    global _PRE
    os.sched_setaffinity(0, cpus)
    import torch
    torch.set_num_threads(1)
    from clipfast.load import load_preprocess_only
    _PRE = (kind, load_preprocess_only())


def _work(blobs):
    import torch
    from PIL import Image
    kind, pre = _PRE
    out = []
    for b in blobs:
        if kind == "pil":
            im = Image.open(io.BytesIO(b)).convert("RGB")
            out.append(pre(im))
        elif kind == "pil_draft":
            im = Image.open(io.BytesIO(b))
            im.draft("RGB", (224 * 2, 224 * 2))    # DCT downscale to >= 2x target while decoding
            out.append(pre(im.convert("RGB")))
        elif kind == "tv":
            import torchvision.io as tio
            import torch.nn.functional as F
            x = tio.decode_jpeg(torch.frombuffer(bytearray(b), dtype=torch.uint8)).float()[None]
            h, w = x.shape[-2:]
            s = 224 / min(h, w)
            nh, nw = max(224, round(h * s)), max(224, round(w * s))
            x = F.interpolate(x, size=(nh, nw), mode="bicubic", antialias=True)
            t, l = (nh - 224) // 2, (nw - 224) // 2
            out.append(x[0, :, t:t + 224, l:l + 224])
    return len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = dict(host=socket.gethostname(), cpus=len(os.sched_getaffinity(0)), job=os.environ.get("CLIPFAST_RUN_ID"))
    try:
        meta["cpu_model"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:  # noqa: BLE001
        pass
    out = open(a.out, "a")

    def rec(**kw):
        kw.update(meta)
        out.write(json.dumps(kw) + "\n")
        out.flush()
        print(json.dumps(kw), flush=True)

    jp = make_jpegs()
    avail = sorted(os.sched_getaffinity(0))
    for kind in ("pil", "pil_draft", "tv"):
        for nw in (1, 4, 8, 16, 24):
            if nw > len(avail):
                continue
            cpus = avail[:nw]
            with mp.get_context("spawn").Pool(nw, initializer=_init, initargs=(kind, cpus)) as pool:
                pool.map(_work, [[jp["512x512"][0]]] * nw)          # warm
                for size, blobs in jp.items():
                    reps = blobs * max(1, (nw * 8) // len(blobs) + 1)
                    chunks = [reps[i::nw * 4] for i in range(nw * 4)]
                    t0 = time.perf_counter()
                    n = sum(pool.map(_work, chunks))
                    dt = time.perf_counter() - t0
                    rec(kind="feed", impl=kind, workers=nw, size=size, imgs=n, img_per_s=n / dt,
                        img_per_s_per_core=n / dt / nw, mb_per_img=sum(len(b) for b in blobs) / len(blobs) / 1e6)
    # tokenizer
    from clipfast.load import load_tokenizer_only
    tok = load_tokenizer_only()
    caps = [("The chest radiograph shows bilateral opacities consistent with pneumonia; figure %d. " % i) * (1 + i % 6)
            for i in range(4096)]
    for par in ("false", "true"):
        os.environ["TOKENIZERS_PARALLELISM"] = par
        t0 = time.perf_counter()
        for i in range(0, len(caps), 1024):
            tok(caps[i:i + 1024], padding="longest", truncation=True, max_length=256)
        dt = time.perf_counter() - t0
        rec(kind="tokenize", impl="hf_fast", parallel=par, captions=len(caps), cap_per_s=len(caps) / dt)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
