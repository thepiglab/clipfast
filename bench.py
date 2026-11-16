"""Benchmark matrix for BiomedCLIP inference: open_clip baselines vs clipfast, plus accuracy vs the fp32 reference.

python bench.py --out results/x.jsonl --impls oc_fp32,oc_amp,oc_compile,fast_eager,fast [--quick]
"""
import argparse
import json
import os
import socket
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clipfast.load import load_reference, probe_images, probe_texts  # noqa: E402
from clipfast.fast import FastBiomedCLIP  # noqa: E402

IMG_B = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
TXT_B = [1, 8, 64, 256, 1024]
TXT_L = [8, 16, 32, 64, 128, 256]
COMPILE_IMG_B = [1, 16, 128, 1024]
COMPILE_TXT_B = [1, 64, 1024]


def timed(fn, min_ms=300.0, reps=5, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    one = (time.perf_counter() - t0) * 1e3
    n = max(1, int(min_ms / max(one, 1e-3)))
    res = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        res.append((time.perf_counter() - t0) * 1e3 / n)
    return statistics.median(res), min(res), max(res)


def pmc_lengths(B, gen):
    """Caption-like token lengths: lognormal, median ~40, clipped to [5, 256]."""
    z = torch.randn(B, generator=gen)
    return (40 * torch.exp(0.8 * z)).round().clamp(5, 256).long()


def text_ids(B, L, vocab, mixed=False, gen=None):
    """[CLS] tok... [SEP] padded with 0 to 256, like open_clip's HF tokenizer output. L="pmc": caption-like lengths."""
    ids = torch.zeros(B, 256, dtype=torch.long)
    pl = pmc_lengths(B, gen) if L == "pmc" else None
    for i in range(B):
        n = int(pl[i]) if pl is not None else (L if not mixed else int(torch.randint(max(3, L // 2), L + 1, (1,), generator=gen)))
        ids[i, 0] = 2
        ids[i, 1:n - 1] = torch.randint(1000, vocab, (n - 2,), generator=gen)
        ids[i, n - 1] = 3
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--impls", default="oc_fp32,oc_amp,oc_compile,fast_eager,fast")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip", default="")
    ap.add_argument("--img_b", default="")
    ap.add_argument("--txt_b", default="")
    ap.add_argument("--txt_l", default="")
    a = ap.parse_args()
    impls = a.impls.split(",")
    skip = set(a.skip.split(",")) if a.skip else set()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = 64
    dev = torch.device("cuda")
    meta = dict(host=socket.gethostname(), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                cuda=torch.version.cuda, job=os.environ.get("CLIPFAST_RUN_ID"))
    print(json.dumps(meta), flush=True)
    out = open(a.out, "a")

    def rec(**kw):
        kw.update(meta)
        out.write(json.dumps(kw) + "\n")
        out.flush()
        print(json.dumps(kw), flush=True)

    ref, preprocess, tok = load_reference()
    vocab = ref.text.config.vocab_size
    gen = torch.Generator().manual_seed(0)

    # ---------- implementations ----------
    MB = 1 << 20
    ref_mb = sum(t.numel() * t.element_size() for t in ref.parameters()) / MB
    torch.cuda.synchronize()
    built0 = torch.cuda.memory_allocated()
    fns = {}
    if "oc_fp32" in impls:
        fns["oc_fp32"] = (lambda x: ref.encode_image(x, normalize=True), lambda t: ref.encode_text(t, normalize=True))
    if "oc_amp" in impls:
        def ai(x):
            with torch.autocast("cuda", torch.bfloat16):
                return ref.encode_image(x, normalize=True)

        def at(t):
            with torch.autocast("cuda", torch.bfloat16):
                return ref.encode_text(t, normalize=True)
        fns["oc_amp"] = (ai, at)
    if "oc_compile" in impls:
        import copy
        mb = copy.deepcopy(ref).to(torch.bfloat16).eval()
        ci = torch.compile(lambda x: mb.encode_image(x.to(torch.bfloat16), normalize=True), mode="max-autotune",
                           dynamic=False)
        ct = torch.compile(lambda t: mb.encode_text(t, normalize=True), mode="max-autotune", dynamic=False)
        fns["oc_compile"] = (ci, ct)
    if "fast_eager" in impls or "fast" in impls:
        fe = FastBiomedCLIP(ref, tok, graphs=False)
        fg = FastBiomedCLIP(ref, tok, graphs=True)
        if "fast_eager" in impls:
            fns["fast_eager"] = (lambda x: fe.encode_image(x), lambda t: fe.encode_text_ids(t))
        if "fast" in impls:
            fns["fast"] = (lambda x: fg.encode_image(x), lambda t: fg.encode_text_ids(t))
    if "fast_c" in impls:
        fc = FastBiomedCLIP(ref, tok, graphs=True, compile=True)
        fns["fast_c"] = (lambda x: fc.encode_image(x), lambda t: fc.encode_text_ids(t))
    if "fast_fused" in impls:
        ff = FastBiomedCLIP(ref, tok, graphs=True, fused=True)
        fns["fast_fused"] = (lambda x: ff.encode_image(x), lambda t: ff.encode_text_ids(t))
    if "fast_fused_pack" in impls:
        ffp = FastBiomedCLIP(ref, tok, graphs=True, fused=True)
        fns["fast_fused_pack"] = (lambda x: ffp.encode_image(x), lambda t: ffp.encode_text_packed(t))
    if "fast_skinny" in impls:
        fsk = FastBiomedCLIP(ref, tok, graphs=True, fused=True, skinny=True)
        fns["fast_skinny"] = (lambda x: fsk.encode_image(x), lambda t: fsk.encode_text_ids(t))
    if "fast_pack" in impls:
        fp = FastBiomedCLIP(ref, tok, graphs=True)
        fns["fast_pack"] = (lambda x: fp.encode_image(x), lambda t: fp.encode_text_packed(t))
    if "fast_ma" in impls:
        fm = FastBiomedCLIP(ref, tok, graphs=True, compile="max-autotune-no-cudagraphs")
        fns["fast_ma"] = (lambda x: fm.encode_image(x), lambda t: fm.encode_text_ids(t))
    if "fast_fp8_ma" in impls:
        f8m = FastBiomedCLIP(ref, tok, graphs=True, compile="max-autotune-no-cudagraphs", fp8=True)
        fns["fast_fp8_ma"] = (lambda x: f8m.encode_image(x), lambda t: f8m.encode_text_ids(t))
    if "fast_fp8" in impls:
        f8 = FastBiomedCLIP(ref, tok, graphs=True, compile=True, fp8=True)
        fns["fast_fp8"] = (lambda x: f8.encode_image(x), lambda t: f8.encode_text_ids(t))
    if "fast_fp32" in impls:              # parity mode: same lean graph, fp32 everywhere, TF32 off
        f32 = FastBiomedCLIP(ref, tok, dtype=torch.float32, graphs=True)
        fns["fast_fp32"] = (lambda x: f32.encode_image(x), lambda t: f32.encode_text_ids(t))
    torch.cuda.synchronize()
    # weights each impl keeps resident: open_clip eager impls use the fp32 reference itself (always loaded here)
    built_mb = (torch.cuda.memory_allocated() - built0) / MB
    weights_mb = {n: (ref_mb if n in ("oc_fp32", "oc_amp") else built_mb / max(1, len([k for k in fns
                  if not k.startswith("oc_") or k == "oc_compile"]))) for n in fns}
    rec(kind="build", impls=list(fns), ref_fp32_weights_mb=ref_mb, built_mb=built_mb, weights_mb=weights_mb)

    def mem_row(fn):
        """Run fn once more with peak tracking: incremental peak over pre-call allocation + whole-device use."""
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        a0 = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return dict(act_peak_mb=(torch.cuda.max_memory_allocated() - a0) / MB,
                    alloc_mb=torch.cuda.memory_allocated() / MB, reserved_mb=torch.cuda.memory_reserved() / MB,
                    device_used_mb=(total - free) / MB)

    # ---------- parity vs open_clip fp32 (TF32 off) ----------
    if "acc" not in skip:
        px, _ = probe_images(preprocess)
        px = px.to(dev)
        labels = ["adenocarcinoma histopathology", "brain MRI", "covid line chart", "squamous cell carcinoma "
                  "histopathology", "immunohistochemistry histopathology", "bone X-ray", "chest X-ray", "pie chart",
                  "hematoxylin and eosin histopathology"]
        tmpl = ["this is a photo of {}", "an image of {}", "{}", "a medical image showing {}",
                "a figure from a paper: {}"]
        texts = probe_texts() + [t.format(l) for t in tmpl for l in labels]
        tids = tok(texts).to(dev)

        def ranks(sim):
            return sim.argsort(1).argsort(1).float()

        def spearman(a, b):
            ra, rb = ranks(a), ranks(b)
            ra, rb = ra - ra.mean(1, keepdim=True), rb - rb.mean(1, keepdim=True)
            return float(((ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1))).mean())

        with torch.inference_mode():
            ri = ref.encode_image(px, normalize=True).float()
            rt = ref.encode_text(tids, normalize=True).float()
            scale = float(ref.logit_scale.exp())
            rs = ri @ rt.T
        for name, (fi, ft) in fns.items():
            with torch.inference_mode():
                ei = torch.cat([fi(px[i:i + 1]).float() for i in range(4)] + [fi(px[4:]).float()])
                et = torch.cat([ft(tids[i:i + 1]).float() for i in range(3)] + [ft(tids[3:]).float()])
                eb_i, eb_t = fi(px).float(), ft(tids).float()      # whole-batch path too
            es = ei @ et.T
            cos_i = torch.minimum((ei * ri).sum(-1), (eb_i * ri).sum(-1))
            cos_t = torch.minimum((et * rt).sum(-1), (eb_t * rt).sum(-1))
            rec(kind="acc", impl=name,
                img_cos_min=float(cos_i.min()), img_cos_mean=float(cos_i.mean()),
                txt_cos_min=float(cos_t.min()), txt_cos_mean=float(cos_t.mean()),
                img_maxabs=float(torch.maximum((ei - ri).abs().max(), (eb_i - ri).abs().max())),
                txt_maxabs=float(torch.maximum((et - rt).abs().max(), (eb_t - rt).abs().max())),
                logit_maxabs=float((scale * (es - rs)).abs().max()),
                i2t_top1_agree=f"{int((es.argmax(1) == rs.argmax(1)).sum())}/{len(ri)}",
                t2i_top1_agree=f"{int((es.argmax(0) == rs.argmax(0)).sum())}/{len(rt)}",
                i2t_top5_overlap=float(sum(len(set(a.tolist()) & set(b.tolist())) for a, b in
                                           zip(es.topk(5, 1).indices, rs.topk(5, 1).indices)) / (5 * len(ri))),
                i2t_rank_spearman=spearman(es, rs), t2i_rank_spearman=spearman(es.T, rs.T),
                zeroshot_agree=f"{int(((es[:9, :9]).argmax(1) == rs[:9, :9].argmax(1)).sum())}/9")

    # ---------- image encoder ----------
    if "image" not in skip:
        for B in ([int(v) for v in a.img_b.split(",")] if a.img_b else (IMG_B if not a.quick else [1, 64, 1024])):
            x = torch.randn(B, 3, 224, 224, device=dev)
            for name, (fi, _) in fns.items():
                if name == "oc_compile" and B not in COMPILE_IMG_B:
                    continue
                with torch.inference_mode():
                    t0 = time.perf_counter()
                    fi(x)
                    torch.cuda.synchronize()
                    first = (time.perf_counter() - t0) * 1e3
                    med, lo, hi = timed(lambda: fi(x))
                    mem = mem_row(lambda: fi(x))
                rec(kind="image", impl=name, B=B, ms=med, ms_min=lo, ms_max=hi, img_per_s=B / med * 1e3,
                    first_call_ms=first, weights_mb=weights_mb[name], **mem)
            del x
            torch.cuda.empty_cache()

    # ---------- text encoder ----------
    if "text" not in skip:
        tl = [int(v) if v.isdigit() else v for v in a.txt_l.split(",")] if a.txt_l else TXT_L + ["mix256", "mix32", "pmc"]
        for B in ([int(v) for v in a.txt_b.split(",")] if a.txt_b else (TXT_B if not a.quick else [1, 64, 1024])):
            for L in tl:
                mixed = isinstance(L, str) and L.startswith("mix")
                Ln = int(L[3:]) if mixed else L
                ids = text_ids(B, Ln, vocab, mixed, gen).to(dev)
                for name, (_, ft) in fns.items():
                    if name.startswith("oc_") and L not in (8, 256, "pmc"):
                        continue          # baselines always run at 256 padded; time does not depend on L
                    if name == "oc_compile" and (B not in COMPILE_TXT_B or L not in (256, "pmc")):
                        continue
                    with torch.inference_mode():
                        med, lo, hi = timed(lambda: ft(ids))
                        mem = mem_row(lambda: ft(ids))
                    rec(kind="text", impl=name, B=B, L=L, ms=med, ms_min=lo, ms_max=hi, txt_per_s=B / med * 1e3,
                        weights_mb=weights_mb[name], **mem)

    # ---------- tokenizer + preprocessing (CPU baselines vs GPU path) ----------
    if "prep" not in skip and "fast" in fns:
        import numpy as np
        from PIL import Image
        for B in [1, 64, 1024]:
            texts = ["this is a photo of chest X-ray number %d with findings" % i for i in range(B)]
            med, _, _ = timed(lambda: tok(texts), min_ms=200)
            rec(kind="tokenize", impl="open_clip_tok_pad256", B=B, ms=med)
            med, _, _ = timed(lambda: fg.tokenize(texts), min_ms=200)
            rec(kind="tokenize", impl="hf_fast_longest", B=B, ms=med)
        for S in [256, 512, 1024, 2048]:
            B = 32
            arrs = [np.random.default_rng(i).integers(0, 255, (S, int(S * 1.25), 3), dtype=np.uint8) for i in range(B)]
            pils = [Image.fromarray(r) for r in arrs]
            med, _, _ = timed(lambda: torch.stack([preprocess(p) for p in pils]), min_ms=300, reps=3, warm=1)
            rec(kind="prep", impl="pil_cpu_1thread", S=S, B=B, ms=med, img_per_s=B / med * 1e3)
            u8 = [torch.from_numpy(r) for r in arrs]
            med, _, _ = timed(lambda: fg.preprocess_gpu(u8), min_ms=300, reps=3, warm=1)
            rec(kind="prep", impl="gpu_from_cpu_u8", S=S, B=B, ms=med, img_per_s=B / med * 1e3)
            u8g = [u.cuda() for u in u8]
            med, _, _ = timed(lambda: fg.preprocess_gpu(u8g), min_ms=300, reps=3, warm=1)
            rec(kind="prep", impl="gpu_from_gpu_u8", S=S, B=B, ms=med, img_per_s=B / med * 1e3)
            with torch.inference_mode():
                a_ = fg.encode_image(torch.stack([preprocess(p) for p in pils[:8]]).cuda())
                b_ = fg.encode_image(fg.preprocess_gpu(u8g[:8]))
            rec(kind="prep_acc", S=S, cos_min=float((a_ * b_).sum(-1).min()))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
