"""Torch convenience wrapper over the native `clipfast_rs` extension (all batching, tokenization and packing run in
Rust; tensors are passed by address, CUDA or CPU).

    from clipfast_rs_torch import ClipFast
    m = ClipFast("rs_assets")                  # fp32=True for parity mode
    img = m.encode_images(pixels)              # float [B,3,224,224] open_clip-preprocessed -> cuda [B,512]
    txt = m.encode_texts(["chest x-ray", ...])  # -> cuda [B,512]
    logits = m.logit_scale * img @ txt.T
"""
import torch

import clipfast_rs


class ClipFast:
    def __init__(self, assets, fp32=False, device="cuda"):
        torch.zeros(1, device=device)          # make the CUDA context current (the engine shares it)
        self.dev = torch.device(device)
        self.eng = clipfast_rs.Engine(assets, fp32)
        self.logit_scale = self.eng.logit_scale

    def encode_images(self, pixels):
        x = pixels.float().contiguous()
        out = torch.empty((x.shape[0], self.eng.dim), device=self.dev)
        torch.cuda.current_stream(self.dev).synchronize()
        self.eng.encode_images_ptr(x.data_ptr(), x.shape[0], out.data_ptr())
        return out

    def encode_texts(self, texts, packed=True):
        out = torch.empty((len(texts), self.eng.dim), device=self.dev)
        torch.cuda.current_stream(self.dev).synchronize()
        self.eng.encode_texts_ptr(list(texts), out.data_ptr(), packed)
        return out

    def encode_token_rows(self, rows, packed=True):
        out = torch.empty((len(rows), self.eng.dim), device=self.dev)
        torch.cuda.current_stream(self.dev).synchronize()
        self.eng.encode_token_rows_ptr([list(map(int, r)) for r in rows], out.data_ptr(), packed)
        return out
