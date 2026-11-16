"""Python binding for the clipfast-rs CUDA engine (ctypes over libclipfast_cuda.so: same kernels, CUDA graphs and
fixed workspace as the Rust CLI). Inputs/outputs are torch tensors; device tensors are passed by pointer (no copy
through the host).

    from clipfast.rs_py import ClipFastRS
    m = ClipFastRS("rs_assets")          # bf16, exact-erf GELU (assets from export_rs.py)
    img = m.encode_images(pixels)        # float [B,3,224,224] open_clip-normalized (cpu or cuda) -> cuda [B,512] L2-normed
    txt = m.encode_texts(["chest x-ray", ...])                          # packed varlen text -> cuda [B,512]
    logits = m.logit_scale * img @ txt.T

fp32=True gives the parity mode (cos 1.0000000 vs open_clip fp32). Not thread-safe: one engine per thread/GPU.
"""
import ctypes
import json
import os

import numpy as np
import torch

# shared library built by scripts/build_lib.sh; override with CLIPFAST_LIB
DEFAULT_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "libclipfast_cuda.so")
MAX_IMG, MAX_ROWS, E, IMG = 256, 65536, 512, 3 * 224 * 224
_LEN_BUCKETS = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256)


def _pow2(n):
    return 1 << (n - 1).bit_length()


def _tok_bucket(t):
    t = max(t, 64)
    step = 1 << max(t.bit_length() - 1 - 3, 0)
    return -(-t // step) * step


class ClipFastRS:
    def __init__(self, assets, lib=None, fp32=False, device=0):
        torch.cuda.set_device(device)
        torch.zeros(1, device="cuda")                  # make torch's primary context current (the engine shares it)
        self.dev = torch.device("cuda", device)
        L = self.lib = ctypes.CDLL(lib or os.environ.get("CLIPFAST_LIB", DEFAULT_LIB))
        vp, ci, sz = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        L.cf_create.restype = vp
        L.cf_create.argtypes = [ci, ci, ci]
        L.cf_set_weight.argtypes = [vp, ctypes.c_char_p, vp, sz]
        L.cf_finalize.argtypes = [vp]
        L.cf_set_option.argtypes = [vp, ctypes.c_char_p, ci]
        L.cf_encode_image.argtypes = [vp, vp, ci, vp, ci]
        L.cf_encode_text.argtypes = [vp, vp, vp, ci, ci, vp, ci]
        L.cf_encode_text_packed.argtypes = [vp, vp, vp, vp, ci, ci, vp, ci]
        L.cf_mem.argtypes = [vp] + [ctypes.POINTER(sz)] * 4
        L.cf_destroy.argtypes = [vp]
        self.cf = L.cf_create(0 if fp32 else 1, MAX_IMG, MAX_ROWS)
        if not self.cf:
            raise RuntimeError("cf_create failed (GPU memory?)")
        from safetensors import safe_open
        path = os.path.join(assets, "weights_%s.safetensors" % ("f32" if fp32 else "bf16"))
        with safe_open(path, "pt") as f:
            for name in f.keys():
                t = f.get_tensor(name).contiguous()
                if L.cf_set_weight(self.cf, name.encode(), t.data_ptr(), t.numel() * t.element_size()) != 0:
                    raise RuntimeError("upload " + name)
        if L.cf_finalize(self.cf) != 0:
            raise RuntimeError("missing weights")
        cfg = json.load(open(os.path.join(assets, "config.json")))
        self.logit_scale = float(cfg["logit_scale"])
        self.pad_id = int(cfg.get("pad_id", 0))
        self._tok = None

    # ---------------- images ----------------
    def encode_images(self, pixels):
        """pixels: float [B,3,224,224], open_clip-preprocessed (normalized). Returns cuda float [B,512], L2-normalized."""
        x = pixels.float().contiguous()
        B = x.shape[0]
        out = torch.empty((B, E), device=self.dev)
        torch.cuda.current_stream(self.dev).synchronize()   # inputs produced on torch's stream are complete
        i = 0
        while i < B:
            n = min(B - i, MAX_IMG)
            nb = _pow2(n)
            src = x[i:i + n]
            if nb != n:
                src = torch.cat([src, src.new_zeros((nb - n,) + tuple(src.shape[1:]))])
            o = out[i:i + n] if nb == n else torch.empty((nb, E), device=self.dev)
            rc = self.lib.cf_encode_image(self.cf, src.data_ptr(), nb, o.data_ptr(), 0)
            if rc != 0:
                raise RuntimeError("cf_encode_image rc=%d" % rc)
            if nb != n:
                out[i:i + n] = o[:n]
            i += n
        return out

    # ---------------- text ----------------
    @property
    def tokenizer(self):
        if self._tok is None:
            from clipfast.load import load_tokenizer_only
            self._tok = load_tokenizer_only()
        return self._tok

    def tokenize(self, texts):
        """open_clip-identical ids (lower-case BERT WordPiece, [CLS] ... [SEP], truncated to 256), no padding."""
        enc = self.tokenizer([" ".join(t.split()) for t in texts], truncation=True, max_length=256,
                             add_special_tokens=True)
        return enc["input_ids"]

    def encode_texts(self, texts, packed=True):
        return self.encode_token_rows(self.tokenize(texts), packed=packed)

    def encode_text_ids(self, ids, packed=True):
        """ids: int tensor [B,L] padded with pad_id (e.g. open_clip tokenizer output)."""
        ids = ids.cpu()
        rows = [r[r != self.pad_id].tolist() for r in ids]
        return self.encode_token_rows(rows, packed=packed)

    def encode_token_rows(self, rows, packed=True):
        B = len(rows)
        out = torch.empty((B, E), device=self.dev)
        rows = [r[:256] for r in rows]
        i = 0
        while i < B:
            if packed:
                n, t = 0, 0
                while i + n < B and n < 1024 and t + len(rows[i + n]) + 1024 <= MAX_ROWS:
                    t += len(rows[i + n])
                    n += 1
                nb = _pow2(n)
                chunk = rows[i:i + n] + [rows[i][:1]] * (nb - n)
                lens = np.fromiter((len(r) for r in chunk), dtype=np.int32, count=nb)
                cu = np.zeros(nb + 1, dtype=np.int32)
                np.cumsum(lens, out=cu[1:])
                tb = _tok_bucket(int(cu[-1]))
                ids = np.zeros(tb, dtype=np.int32)
                ids[:cu[-1]] = np.fromiter((v for r in chunk for v in r), dtype=np.int32, count=int(cu[-1]))
                pos = np.zeros(tb, dtype=np.int32)
                pos[:cu[-1]] = np.arange(cu[-1], dtype=np.int32) - np.repeat(cu[:-1], lens)
                o = torch.empty((nb, E), device=self.dev)
                rc = self.lib.cf_encode_text_packed(self.cf, ids.ctypes.data, pos.ctypes.data, cu.ctypes.data, nb, tb,
                                                    o.data_ptr(), 0)
            else:
                lmax = max(len(r) for r in rows[i:i + 1024])
                L = next(b for b in _LEN_BUCKETS if lmax <= b)
                n = min(B - i, 1024, MAX_ROWS // L)
                L = next(b for b in _LEN_BUCKETS if max(len(r) for r in rows[i:i + n]) <= b)
                nb = min(_pow2(n), MAX_ROWS // L)
                ids = np.full((nb, L), self.pad_id, dtype=np.int32)
                lens = np.ones(nb, dtype=np.int32)
                for j, r in enumerate(rows[i:i + n]):
                    ids[j, :len(r)] = r
                    lens[j] = len(r)
                ids[n:, 0] = rows[i][0]
                o = torch.empty((nb, E), device=self.dev)
                rc = self.lib.cf_encode_text(self.cf, ids.ctypes.data, lens.ctypes.data, nb, L, o.data_ptr(), 0)
            if rc != 0:
                raise RuntimeError("encode text rc=%d" % rc)
            out[i:i + n] = o[:n]
            i += n
        return out

    def memory_mb(self):
        v = [ctypes.c_size_t() for _ in range(4)]
        self.lib.cf_mem(self.cf, *[ctypes.byref(x) for x in v])
        return dict(weights_mb=v[0].value / 2**20, workspace_mb=v[1].value / 2**20, device_used_mb=v[2].value / 2**20)

    def __del__(self):
        if getattr(self, "cf", None):
            self.lib.cf_destroy(self.cf)
            self.cf = None
