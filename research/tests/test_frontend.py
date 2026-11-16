"""Parity test for the Rust front end (clipfast-rs prep / tokenize / embed) against open_clip's Python path:
  1. pixels: PIL Image.open().convert("RGB") + open_clip eval transform vs `clipfast-rs prep` on the same files
     (JPEG at several sizes/aspects/qualities incl. grayscale and 4:4:4/4:2:0 chroma, PNG RGB/RGBA/gray/palette);
  2. token ids: open_clip tokenizer vs `clipfast-rs tokenize`;
  3. end to end: `clipfast-rs embed` embeddings vs open_clip fp32 on the same files / captions.
python test_frontend.py BIN ASSETS WORKDIR OUT.jsonl
"""
import io
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clipfast.load import load_reference, probe_texts, MODEL_DIR  # noqa: E402


def make_files(d):
    from PIL import Image
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    ex = os.path.join(MODEL_DIR, "example_data", "biomed_image_classification_example_data")
    for f in sorted(os.listdir(ex)):
        paths.append(os.path.join(ex, f))
    sizes = [(224, 224), (225, 301), (300, 225), (512, 512), (640, 480), (1023, 767), (2048, 1536), (97, 333),
             (3000, 2400), (226, 226), (1600, 223)]
    for i, (w, h) in enumerate(sizes):
        base = rng.random((h // 16 + 2, w // 16 + 2, 3))
        im = Image.fromarray((base * 255).astype("uint8")).resize((w, h), Image.BICUBIC)
        arr = (np.asarray(im).astype(np.int16) + rng.integers(-20, 20, (h, w, 3))).clip(0, 255).astype("uint8")
        im = Image.fromarray(arr)
        for q, sub in ((92, 0), (75, 2)):
            p = os.path.join(d, f"syn{i}_{w}x{h}_q{q}_s{sub}.jpg")
            im.save(p, format="JPEG", quality=q, subsampling=sub)
            paths.append(p)
        if i % 3 == 0:
            p = os.path.join(d, f"syn{i}_{w}x{h}.png")
            im.save(p)
            paths.append(p)
            p = os.path.join(d, f"syn{i}_{w}x{h}_gray.jpg")
            im.convert("L").save(p, quality=90)
            paths.append(p)
            p = os.path.join(d, f"syn{i}_{w}x{h}_rgba.png")
            im.convert("RGBA").save(p)
            paths.append(p)
            p = os.path.join(d, f"syn{i}_{w}x{h}_pal.png")
            im.convert("P").save(p)
            paths.append(p)
    lst = os.path.join(d, "list.txt")
    open(lst, "w").write("\n".join(paths))
    return paths, lst


def load_npy(p):
    return np.load(p)


def main(binp, assets, work, out_path):
    out = open(out_path, "a")

    def rec(**kw):
        out.write(json.dumps(kw) + "\n")
        out.flush()
        print(json.dumps(kw), flush=True)

    from PIL import Image
    ref, pre, tok = load_reference()
    paths, lst = make_files(os.path.join(work, "imgs"))
    # 1. pixels
    subprocess.run([binp, "prep", "--images", lst, "--out", os.path.join(work, "rs_px.npy")], check=True)
    rs_px = load_npy(os.path.join(work, "rs_px.npy"))
    rs_names = open(os.path.join(work, "rs_px.npy.txt")).read().splitlines()
    worst, n_exact = 0.0, 0
    per = []
    for p, x in zip(rs_names, rs_px):
        py = pre(Image.open(p).convert("RGB")).numpy()
        d = float(np.abs(py - x).max())
        per.append((os.path.basename(p), d))
        worst = max(worst, d)
        n_exact += d == 0.0
    one_level = 1 / 255 / 0.26130258 + 1e-6      # one uint8 step after normalization (largest 1/std)
    rec(kind="frontend_pixels", files=len(rs_names), requested=len(paths), bit_exact=n_exact,
        max_abs=worst, max_abs_in_uint8_steps=worst / (1 / 255 / 0.26862954),
        worst=sorted(per, key=lambda t: -t[1])[:5], within_one_level=sum(d <= one_level for _, d in per))
    # 2. tokens
    texts = probe_texts() + ["  multiple   spaces\tand\ttabs  ", "UPPER case & symbols: <b>bold</b> 5mg/kg, 37°C",
                             "naïve café résumé — em-dash … ellipsis", "x" * 2000]
    tf = os.path.join(work, "texts.txt")
    open(tf, "w").write("\n".join(texts))
    r = subprocess.run([binp, "tokenize", "--assets", assets, "--texts", tf], check=True, capture_output=True, text=True)
    rs_ids = [json.loads(l) for l in r.stdout.splitlines()]
    oc = tok(texts)
    oc_ids = [row[row != 0].tolist() for row in oc]
    mism = [i for i, (a, b) in enumerate(zip(rs_ids, oc_ids)) if a != b]
    rec(kind="frontend_tokens", texts=len(texts), match=len(texts) - len(mism), mismatches=[texts[i][:60] for i in mism])
    # 3. end to end embeddings vs open_clip fp32
    for mode in ([], ["--fp32"]):
        prefix = os.path.join(work, "emb" + ("_fp32" if mode else ""))
        r = subprocess.run([binp, "embed", "--assets", assets, "--images", lst, "--texts", tf, "--out", prefix] + mode,
                           check=True, capture_output=True, text=True)
        summary = json.loads(r.stdout.strip().splitlines()[-1])
        ei = torch.from_numpy(load_npy(prefix + ".images.npy"))
        names = open(prefix + ".images.txt").read().splitlines()
        et = torch.from_numpy(load_npy(prefix + ".texts.npy"))
        with torch.inference_mode():
            px = torch.stack([pre(Image.open(p).convert("RGB")) for p in names]).cuda()
            ri = ref.encode_image(px, normalize=True).float().cpu()
            rt = ref.encode_text(oc.cuda(), normalize=True).float().cpu()
        rec(kind="frontend_embed", impl=summary["impl"], images=len(names), img_cos_min=float((ei * ri).sum(-1).min()),
            txt_cos_min=float((et * rt).sum(-1).min()),
            i2t_top1_agree=f"{int(((ei @ et.T).argmax(1) == (ri @ rt.T).argmax(1)).sum())}/{len(names)}",
            summary=summary)


if __name__ == "__main__":
    main(*sys.argv[1:5])
