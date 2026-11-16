"""clipfast v0: a lean BiomedCLIP inference runtime (bf16) built from the reference open_clip weights.

Structural changes vs open_clip (each checked against the fp32 reference by bench.py):
  - bf16 weights, fp32 accumulation inside cuBLAS/cuDNN; fused QKV for BERT; SDPA (flash / cuDNN) attention.
  - Last layer of each tower computes only the CLS query/MLP (the only row the embedding uses).
  - Text: no padding to 256; batch padded to the next length bucket, key-padding mask only when lengths differ.
  - CUDA graphs per (tower, batch bucket, length bucket); batches padded up to power-of-two buckets.
  - Optional GPU image preprocessing (resize shortest side, center crop, normalize) from uint8 tensors.
"""
import math

import torch
import torch.nn.functional as F

BATCH_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
LEN_BUCKETS = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def _bucket(n, buckets):
    for b in buckets:
        if n <= b:
            return b
    return None


class FastBiomedCLIP:
    def __init__(self, ref, tokenizer=None, dtype=torch.bfloat16, graphs=True, compile=False, fp8=False, fused=False,
                 skinny=False):
        self.dtype, self.graphs, self.dev = dtype, graphs, next(ref.parameters()).device
        self.fp8, self.fused, self.skinny = fp8, fused, skinny
        if skinny:
            from clipfast.kernels import skinny_linear
            self._sk = skinny_linear
        if fused:
            from clipfast.kernels import fused_linear
            self._fl = fused_linear
        g = lambda t: t.detach().to(self.dev, dtype)
        v, t = ref.visual, ref.text
        tr = v.trunk
        assert tr.global_pool == "token" and tr.cls_token is not None and tr.pos_embed.shape[1] == 197
        self.v_patch_w, self.v_patch_b = g(tr.patch_embed.proj.weight), g(tr.patch_embed.proj.bias)
        self.v_cls_pos0 = g(tr.cls_token[0, 0] + tr.pos_embed[0, 0])
        self.v_pos = g(tr.pos_embed[0, 1:])
        self.v_blocks = []
        for b in tr.blocks:
            assert b.norm1.eps == 1e-6 and isinstance(b.mlp.act, torch.nn.GELU)
            self.v_blocks.append(dict(
                n1w=g(b.norm1.weight), n1b=g(b.norm1.bias), qkv_w=g(b.attn.qkv.weight), qkv_b=g(b.attn.qkv.bias),
                o_w=g(b.attn.proj.weight), o_b=g(b.attn.proj.bias), n2w=g(b.norm2.weight), n2b=g(b.norm2.bias),
                f1_w=g(b.mlp.fc1.weight), f1_b=g(b.mlp.fc1.bias), f2_w=g(b.mlp.fc2.weight), f2_b=g(b.mlp.fc2.bias)))
        self.v_nw, self.v_nb = g(tr.norm.weight), g(tr.norm.bias)
        self.v_proj = g(v.head.proj.weight)
        self.v_heads = tr.blocks[0].attn.num_heads

        bert = t.transformer
        e = bert.embeddings
        self.t_word, self.t_pos = g(e.word_embeddings.weight), g(e.position_embeddings.weight)
        self.t_type0 = g(e.token_type_embeddings.weight[0])
        self.t_lnw, self.t_lnb, self.t_eps = g(e.LayerNorm.weight), g(e.LayerNorm.bias), e.LayerNorm.eps
        self.t_blocks = []
        for L in bert.encoder.layer:
            a = L.attention
            self.t_blocks.append(dict(
                qkv_w=g(torch.cat([a.self.query.weight, a.self.key.weight, a.self.value.weight])),
                qkv_b=g(torch.cat([a.self.query.bias, a.self.key.bias, a.self.value.bias])),
                o_w=g(a.output.dense.weight), o_b=g(a.output.dense.bias),
                n1w=g(a.output.LayerNorm.weight), n1b=g(a.output.LayerNorm.bias),
                f1_w=g(L.intermediate.dense.weight), f1_b=g(L.intermediate.dense.bias),
                f2_w=g(L.output.dense.weight), f2_b=g(L.output.dense.bias),
                n2w=g(L.output.LayerNorm.weight), n2b=g(L.output.LayerNorm.bias)))
        self.t_heads = bert.config.num_attention_heads
        self.t_p1, self.t_p2 = g(t.proj[0].weight), g(t.proj[2].weight)
        self.pad_id = t.config.pad_token_id or 0
        self.logit_scale = ref.logit_scale.detach().exp().float()
        pc = getattr(ref.visual, "preprocess_cfg", None) or {}
        mean = pc.get("mean", (0.48145466, 0.4578275, 0.40821073))
        std = pc.get("std", (0.26862954, 0.26130258, 0.27577711))
        self.mean = torch.tensor(mean, device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.dev).view(1, 3, 1, 1)
        self.hf_tok = getattr(tokenizer, "tokenizer", None)
        self._graphs = {}
        self._pool = torch.cuda.graph_pool_handle() if graphs else None   # ONE pool shared by every bucket graph
        if fp8:
            for p in self.v_blocks + self.t_blocks:
                for k in ("qkv", "o", "f1", "f2"):
                    w = p[k + "_w"].float()
                    sw = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0
                    p[k + "_q"] = ((w / sw).to(torch.float8_e4m3fn).t(), sw.t().contiguous())
        self._vision, self._text = self._vision_impl, self._text_impl
        if compile:                     # True -> inductor default; or a mode string, e.g. "max-autotune-no-cudagraphs"
            mode = None if compile is True else compile
            self._vision = torch.compile(self._vision_impl, dynamic=False, mode=mode)
            self._text = torch.compile(self._text_impl, dynamic=False, mode=mode)

    def _lin(self, x, p, k, mode=0, r=None):
        """Linear for the big per-block weights: bf16 cuBLAS, Triton with fused epilogue (mode 1 GELU, 2 +r),
        or fp8 e4m3 with rowwise scales (opt-in). Returns the epilogue result."""
        rows = x.numel() // x.shape[-1]
        # kar A/B (fusedlin_ab1): fused Triton epilogues win for GELU/residual at M >= ~25k rows (+2..+71%) and lose
        # on plain bias GEMMs and small M, where cuBLAS stays.
        if self.fused and not self.fp8 and mode in (1, 2) and rows >= 16384:
            return self._fl(x, p[k + "_w"], p[k + "_b"], r, mode)
        y = self._lin0(x, p, k)
        if mode == 1:
            return F.gelu(y)
        if mode == 2:
            return r + y
        return y

    def _lin0(self, x, p, k):
        # kar A/B (skinny_ab1, RTX PRO 6000): v1d wins or ties only for K == 768 and M <= 16; cuBLAS elsewhere
        if self.skinny and not self.fp8 and x.shape[-1] == 768 and x.numel() // x.shape[-1] <= 16:
            shp = x.shape
            y = self._sk(x.reshape(-1, shp[-1]).contiguous(), p[k + "_w"], p[k + "_b"])
            return y.view(*shp[:-1], y.shape[-1])
        if not self.fp8:
            return F.linear(x, p[k + "_w"], p[k + "_b"])
        wq, sw = p[k + "_q"]
        shp = x.shape
        a = x.reshape(-1, shp[-1]).float()
        sa = a.abs().amax(1, keepdim=True).clamp_min(1e-12) / 448.0
        aq = (a / sa).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(aq, wq, scale_a=sa, scale_b=sw, bias=p[k + "_b"], out_dtype=self.dtype)
        return y.view(*shp[:-1], y.shape[-1])

    # ---------------- towers (eager, graph-capturable) ----------------
    def _vision_impl(self, x):
        B = x.shape[0]
        H = self.v_heads
        x = F.conv2d(x.to(self.dtype), self.v_patch_w, self.v_patch_b, stride=16)       # B,768,14,14
        x = x.flatten(2).transpose(1, 2) + self.v_pos                                  # B,196,768
        x = torch.cat([self.v_cls_pos0.expand(B, 1, -1), x], 1)                        # B,197,768
        N, C = x.shape[1], x.shape[2]
        nb = len(self.v_blocks)
        for i, p in enumerate(self.v_blocks):
            h = F.layer_norm(x, (C,), p["n1w"], p["n1b"], 1e-6)
            qkv = self._lin(h, p, "qkv").view(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            if i == nb - 1:                                    # only the CLS row is used downstream
                q = q[:, :, :1]
                x = x[:, :1]
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, q.shape[2], C)
            x = self._lin(a, p, "o", 2, x)
            h = F.layer_norm(x, (C,), p["n2w"], p["n2b"], 1e-6)
            x = self._lin(self._lin(h, p, "f1", 1), p, "f2", 2, x)
        x = F.layer_norm(x[:, 0], (C,), self.v_nw, self.v_nb, 1e-6)
        return F.linear(x, self.v_proj).float()

    def _text_impl(self, ids, mask):
        """ids [B,L] int64, mask None or bool [B,L] (True = real token)."""
        B, L = ids.shape
        H = self.t_heads
        x = self.t_word[ids] + self.t_pos[:L] + self.t_type0
        C = x.shape[-1]
        x = F.layer_norm(x, (C,), self.t_lnw, self.t_lnb, self.t_eps)
        am = None if mask is None else mask[:, None, None, :]
        nb = len(self.t_blocks)
        for i, p in enumerate(self.t_blocks):
            qkv = self._lin(x, p, "qkv").view(B, L, 3, H, C // H).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            if i == nb - 1:
                q = q[:, :, :1]
                x = x[:, :1]
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=am).transpose(1, 2).reshape(B, q.shape[2], C)
            x = F.layer_norm(self._lin(a, p, "o", 2, x), (C,), p["n1w"], p["n1b"], self.t_eps)
            x = F.layer_norm(self._lin(self._lin(x, p, "f1", 1), p, "f2", 2, x), (C,), p["n2w"], p["n2b"], self.t_eps)
        x = x[:, 0]
        return F.linear(F.gelu(F.linear(x, self.t_p1)), self.t_p2).float()

    def _varlen_attn(self, q, k, v, cq, ck, mq, mk):
        """q [Tq,H,D], k/v [Tk,H,D] packed; cq/ck int32 cumulative lengths [B+1]. Flash varlen forward."""
        out = torch.ops.aten._flash_attention_forward(q, k, v, cq, ck, mq, mk, 0.0, False, False)
        return out[0]

    def _text_packed_impl(self, tok, pos, cu, cls_idx, cu_cls, max_len):
        """Packed (no padding) text tower. tok/pos [T] int64, cu int32 [B+1], cls_idx int64 [B]."""
        H = self.t_heads
        x = self.t_word[tok] + self.t_pos[pos] + self.t_type0
        C = x.shape[-1]
        D = C // H
        x = F.layer_norm(x, (C,), self.t_lnw, self.t_lnb, self.t_eps)
        nb = len(self.t_blocks)
        for i, p in enumerate(self.t_blocks):
            qkv = self._lin(x, p, "qkv").view(-1, 3, H, D)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            if i == nb - 1:
                q, x = q[cls_idx], x[cls_idx]
                a = self._varlen_attn(q.contiguous(), k, v, cu_cls, cu, 1, max_len)
            else:
                a = self._varlen_attn(q, k, v, cu, cu, max_len, max_len)
            a = a.reshape(-1, C)
            x = F.layer_norm(self._lin(a, p, "o", 2, x), (C,), p["n1w"], p["n1b"], self.t_eps)
            x = F.layer_norm(self._lin(self._lin(x, p, "f1", 1), p, "f2", 2, x), (C,), p["n2w"], p["n2b"], self.t_eps)
        return F.linear(F.gelu(F.linear(x, self.t_p1)), self.t_p2).float()

    @torch.inference_mode()
    def encode_text_packed(self, ids, normalize=True):
        """Variable-length path for large mixed-length batches: only real tokens are computed (no padding).
        ids: int [B,L] padded with pad_id (CPU preferred). No CUDA graph (shapes vary per batch)."""
        ids = ids.cpu()
        mask = ids != self.pad_id
        lens = mask.sum(1)
        L = ids.shape[1]
        tok = ids[mask]
        pos = torch.arange(L).expand_as(ids)[mask]
        cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(lens, 0)
        cls_idx = cu[:-1].long()
        cu_cls = torch.arange(len(lens) + 1, dtype=torch.int32)
        d = self.dev
        y = self._text_packed_impl(tok.to(d, non_blocking=True), pos.to(d, non_blocking=True),
                                   cu.to(d, non_blocking=True), cls_idx.to(d, non_blocking=True),
                                   cu_cls.to(d, non_blocking=True), int(lens.max()))
        return F.normalize(y, dim=-1) if normalize else y

    # ---------------- graph cache ----------------
    def _graph(self, key, fn, static):
        """Capture fn(*static) once per key; later calls refill the static buffers and replay."""
        ent = self._graphs.get(key)
        if ent is None:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    fn(*static)
            torch.cuda.current_stream().wait_stream(s)
            gr = torch.cuda.CUDAGraph()
            # Shared pool: graphs replay one at a time and outputs are copied out right after each replay, and the
            # static inputs/outputs of every graph stay referenced here, so no live buffer is ever reused.
            with torch.cuda.graph(gr, pool=self._pool):
                out = fn(*static)
            ent = self._graphs[key] = (gr, static, out)
        return ent

    def _vis_out(self, x, normalize):
        y = self._vision(x)
        return F.normalize(y, dim=-1) if normalize else y

    def _txt_out(self, ids, masked, normalize):
        y = self._text(ids, (ids != self.pad_id) if masked else None)
        return F.normalize(y, dim=-1) if normalize else y

    # ---------------- public API ----------------
    @torch.inference_mode()
    def encode_image(self, pixels, normalize=True):
        """pixels: float [B,3,224,224] already normalized (open_clip preprocess output), any device."""
        outs = []
        for chunk in pixels.split(BATCH_BUCKETS[-1]):
            B = chunk.shape[0]
            if self.graphs:
                Bb = _bucket(B, BATCH_BUCKETS)
                key = ("v", Bb, normalize)
                ent = self._graphs.get(key)
                if ent is None:
                    st = torch.zeros((Bb,) + tuple(chunk.shape[1:]), device=self.dev, dtype=self.dtype)
                    ent = self._graph(key, lambda x: self._vis_out(x, normalize), [st])
                gr, (st,), out = ent
                st[:B].copy_(chunk, non_blocking=True)     # rows >= B keep stale finite data; their output is dropped
                gr.replay()
                outs.append(out[:B].clone())
            else:
                outs.append(self._vis_out(chunk.to(self.dev, non_blocking=True), normalize))
        return outs[0] if len(outs) == 1 else torch.cat(outs)

    def tokenize(self, texts, context_length=256):
        enc = self.hf_tok(texts, padding="longest", truncation=True, max_length=context_length,
                          return_tensors="pt", return_attention_mask=False)
        return enc["input_ids"]

    @torch.inference_mode()
    def encode_text_ids(self, ids, normalize=True):
        """ids: int [B,L] padded with pad_id (any L <= 256; trailing pads are trimmed). CPU ids avoid a device sync."""
        lens = (ids != self.pad_id).sum(1)
        outs = []
        for chunk, lc in zip(ids.split(BATCH_BUCKETS[-1]), lens.split(BATCH_BUCKETS[-1])):
            B = chunk.shape[0]
            Lmax, Lmin = int(lc.max()), int(lc.min())
            if self.graphs:
                Lb, Bb = _bucket(Lmax, LEN_BUCKETS), _bucket(B, BATCH_BUCKETS)
                masked = not (Lmin == Lb and Lmax == Lb)
                key = ("t", Bb, Lb, masked, normalize)
                ent = self._graphs.get(key)
                if ent is None:
                    st = torch.full((Bb, Lb), self.pad_id, dtype=torch.long, device=self.dev)
                    st[:, 0] = 2
                    ent = self._graph(key, lambda t: self._txt_out(t, masked, normalize), [st])
                gr, (st,), out = ent
                w = min(Lb, chunk.shape[1])
                x = torch.full((Bb, Lb), self.pad_id, dtype=torch.long, device=chunk.device)
                x[:B, :w] = chunk[:, :w]
                x[B:, :] = x[:1, :]                          # dummy rows copy row 0 (never fully masked)
                st.copy_(x, non_blocking=True)
                gr.replay()
                outs.append(out[:B].clone())
            else:
                x = chunk[:, :Lmax].to(self.dev, non_blocking=True)
                outs.append(self._txt_out(x, Lmin != Lmax, normalize))
        return outs[0] if len(outs) == 1 else torch.cat(outs)

    def encode_text(self, texts, normalize=True):
        return self.encode_text_ids(self.tokenize(texts), normalize)

    @torch.inference_mode()
    def preprocess_gpu(self, images_u8, size=224):
        """images_u8: list of uint8 [H,W,3] tensors (CPU or GPU). Resize shortest side (bicubic, antialias),
        center crop, normalize. Returns float [B,3,size,size] on device."""
        out = torch.empty((len(images_u8), 3, size, size), device=self.dev, dtype=torch.float32)
        for i, im in enumerate(images_u8):
            x = im.to(self.dev, non_blocking=True).permute(2, 0, 1)[None].float()
            h, w = x.shape[-2:]
            s = size / min(h, w)
            nh, nw = max(size, round(h * s)), max(size, round(w * s))
            x = F.interpolate(x, size=(nh, nw), mode="bicubic", antialias=True, align_corners=False)
            t, l = (nh - size) // 2, (nw - size) // 2
            out[i] = ((x[0, :, t:t + size, l:l + size].clamp(0, 255) / 255.0) - self.mean[0]) / self.std[0]
        return out
