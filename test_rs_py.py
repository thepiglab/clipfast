"""Test the Python binding of the clipfast-rs engine: tokenizer ids vs open_clip, embedding parity vs the open_clip
fp32 golden set (bf16 + fp32 modes, cuda + cpu inputs, packed + padded text), and speed from Python.
python test_rs_py.py ASSETS OUT.jsonl
"""
import json
import os
import socket
import sys
import time

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clipfast.rs_py import ClipFastRS  # noqa: E402
from bench import pmc_lengths  # noqa: E402


def main(assets, out_path):
    out = open(out_path, "a")
    meta = dict(host=socket.gethostname(), gpu=torch.cuda.get_device_name(), job=os.environ.get("CLIPFAST_RUN_ID"))

    def rec(**kw):
        kw.update(meta)
        out.write(json.dumps(kw) + "\n")
        out.flush()
        print(json.dumps(kw), flush=True)

    g = load_file(os.path.join(assets, "golden.safetensors"))
    texts = json.load(open(os.path.join(assets, "golden_texts.json")))["texts"]
    px, ids, ri, rt = g["pixels"], g["ids"], g["img_emb"].cuda(), g["txt_emb"].cuda()
    for fp32 in (False, True):
        t0 = time.perf_counter()
        m = ClipFastRS(assets, fp32=fp32)
        load_s = time.perf_counter() - t0
        name = "rs_py_fp32" if fp32 else "rs_py_bf16"
        golden_rows = [r[r != 0].tolist() for r in ids]
        ours = m.tokenize(texts)
        tok_match = sum(a == b for a, b in zip(ours, golden_rows))
        ei_cuda = m.encode_images(px.cuda())
        ei_cpu = m.encode_images(px)
        et_pack = m.encode_texts(texts, packed=True)
        et_pad = m.encode_texts(texts, packed=False)
        et_ids = m.encode_text_ids(ids)
        cos = lambda a, b: float((a * b).sum(-1).min())
        es, rs = ei_cuda @ et_pack.T, ri @ rt.T
        rec(kind="acc", impl=name, tokenizer_ids_match=f"{tok_match}/{len(texts)}",
            img_cos_min=cos(ei_cuda, ri), img_cos_min_cpu_input=cos(ei_cpu, ri),
            txt_cos_min_packed=cos(et_pack, rt), txt_cos_min_padded=cos(et_pad, rt), txt_cos_min_from_ids=cos(et_ids, rt),
            logit_maxabs=float((m.logit_scale * (es - rs)).abs().max()),
            i2t_top1_agree=f"{int((es.argmax(1) == rs.argmax(1)).sum())}/{len(ri)}",
            t2i_top1_agree=f"{int((es.argmax(0) == rs.argmax(0)).sum())}/{len(rt)}",
            zeroshot_agree=f"{int((es[:9, :9].argmax(1) == rs[:9, :9].argmax(1)).sum())}/9", load_s=load_s,
            **m.memory_mb(), torch_alloc_mb=torch.cuda.memory_allocated() / 2**20)
        if fp32:
            break
        # speed from Python (wrapper syncs; inputs resident on GPU)
        for B in (1, 64, 1024):
            x = torch.randn(B, 3, 224, 224, device="cuda")
            for _ in range(3):
                m.encode_images(x)
            n = max(1, int(2000 / B))
            t0 = time.perf_counter()
            for _ in range(n):
                m.encode_images(x)
            dt = (time.perf_counter() - t0) / n
            rec(kind="image", impl=name, B=B, ms=dt * 1e3, img_per_s=B / dt, **m.memory_mb())
        gen = torch.Generator().manual_seed(0)
        lens = pmc_lengths(1024, gen)
        rows = [[2] + torch.randint(1000, 30000, (int(l) - 2,), generator=gen).tolist() + [3] for l in lens]
        caps = [" ".join(["pneumonia"] * max(1, int(l) - 2)) for l in lens]
        for label, fn in (("ids_pmc", lambda: m.encode_token_rows(rows)),
                          ("strings_pmc_incl_tokenizer", lambda: m.encode_texts(caps))):
            for _ in range(3):
                fn()
            t0 = time.perf_counter()
            for _ in range(5):
                fn()
            dt = (time.perf_counter() - t0) / 5
            rec(kind="text", impl=name, B=1024, L=label, ms=dt * 1e3, txt_per_s=1024 / dt, **m.memory_mb())
        del m
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
