"""Turn bench/train jsonl results into markdown tables.  python report.py results/*.jsonl > REPORT.md"""
import json
import sys
from collections import defaultdict

BASES = ["oc_amp", "oc_compile"]


def load(paths):
    rows = []
    for p in paths:
        for line in open(p):
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def latest(rows, kind, keyf):
    d = {}
    for r in rows:
        if r.get("kind") == kind and not r.get("oom"):
            d[keyf(r)] = r              # later files / rows win
    return d


def main(paths):
    rows = load(paths)
    out = []
    acc = latest(rows, "acc", lambda r: r["impl"])
    if acc:
        out += ["## Accuracy vs open_clip fp32 (min cosine over probe set, zero-shot agreement)", "",
                "| impl | image cos min | text cos min | zero-shot |", "|---|---|---|---|"]
        for k, r in acc.items():
            out.append(f"| {k} | {r['img_cos_min']:.5f} | {r['txt_cos_min']:.5f} | {r['zeroshot_agree']} |")
        out.append("")

    img = latest(rows, "image", lambda r: (r["impl"], r["B"]))
    if img:
        impls = sorted({k[0] for k in img}, key=lambda s: (not s.startswith("oc"), s))
        Bs = sorted({k[1] for k in img})
        out += ["## Image encoder throughput (img/s; higher is better)", "",
                "| B | " + " | ".join(impls) + " | best fast / oc_compile |", "|---" * (len(impls) + 2) + "|"]
        for B in Bs:
            cells, best = [], None
            for i in impls:
                r = img.get((i, B))
                cells.append(f"{r['img_per_s']:.0f}" if r else "")
                if r and not i.startswith("oc") and i != "fast_fp8" and "fp8" not in i:
                    best = max(best or 0, r["img_per_s"])
            base = img.get(("oc_compile", B)) or img.get(("oc_amp", B))
            sp = f"{best / base['img_per_s']:.2f}x" if best and base else ""
            out.append(f"| {B} | " + " | ".join(cells) + f" | {sp} |")
        out.append("")

    txt = latest(rows, "text", lambda r: (r["impl"], r["B"], str(r["L"])))
    if txt:
        impls = sorted({k[0] for k in txt}, key=lambda s: (not s.startswith("oc"), s))
        keys = sorted({(k[1], k[2]) for k in txt}, key=lambda t: (t[0], (0, int(t[1])) if t[1].isdigit() else (1, t[1])))
        out += ["## Text encoder latency (ms per batch; lower is better). open_clip always pads to 256 tokens,",
                "so its time at any L equals its L=256 time.", "",
                "| B | L | " + " | ".join(impls) + " |", "|---" * (len(impls) + 2) + "|"]
        for B, L in keys:
            cells = []
            for i in impls:
                r = txt.get((i, B, L))
                if r is None and i.startswith("oc"):
                    r = txt.get((i, B, "256"))
                cells.append(f"{r['ms']:.3f}" if r else "")
            out.append(f"| {B} | {L} | " + " | ".join(cells) + " |")
        out.append("")

    tr = latest(rows, "train", lambda r: (r["impl"], r["B"], str(r["L"])))
    if tr:
        impls = sorted({k[0] for k in tr}, key=lambda s: (not s.startswith("oc"), s))
        keys = sorted({(k[1], k[2]) for k in tr})
        out += ["## Training step throughput (samples/s, fwd+bwd+AdamW; higher is better)", "",
                "| B | L | " + " | ".join(impls) + " |", "|---" * (len(impls) + 2) + "|"]
        for B, L in keys:
            out.append(f"| {B} | {L} | " + " | ".join(
                f"{tr[(i, B, L)]['samples_per_s']:.0f}" if (i, B, L) in tr else "" for i in impls) + " |")
        out.append("")
    for kind in ("train_gradcheck", "train_parity", "prep", "prep_acc", "tokenize"):
        rs = [r for r in rows if r.get("kind") == kind]
        if rs:
            out += [f"## {kind}", ""]
            for r in rs:
                out.append("- " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()
                                            if k not in ("host", "gpu", "torch", "cuda", "job", "kind")))
            out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1:])
