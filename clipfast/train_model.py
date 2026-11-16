"""Trainable lean BiomedCLIP for contrastive fine-tuning (same math as open_clip's model, fewer wasted FLOPs).

- fp32 master parameters, bf16 autocast compute (like open_clip --precision amp_bf16).
- Last block of each tower computes only the CLS row's query/MLP: the loss only reads CLS, so this is exact.
- Text: packed variable-length tokens (no padding) with flash varlen attention (has autograd), or padded+mask SDPA.
- Dropout is off (BiomedCLIP's configs use 0 for the towers anyway; open_clip HF BERT default 0.1 hidden dropout
  is disabled by the caller in train_bench for a like-for-like comparison).
State dict keys mirror open_clip's so checkpoints can be copied back with `export_to(ref_model)`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _P(t):
    return nn.Parameter(t.detach().clone().float())


class LeanBiomedCLIP(nn.Module):
    def __init__(self, ref):
        super().__init__()
        tr, v, t = ref.visual.trunk, ref.visual, ref.text
        self.v_patch_w, self.v_patch_b = _P(tr.patch_embed.proj.weight), _P(tr.patch_embed.proj.bias)
        self.v_cls, self.v_pos = _P(tr.cls_token), _P(tr.pos_embed)
        self.v_blocks = nn.ModuleList()
        for b in tr.blocks:
            m = nn.Module()
            for name, src in dict(n1w=b.norm1.weight, n1b=b.norm1.bias, qkv_w=b.attn.qkv.weight, qkv_b=b.attn.qkv.bias,
                                  o_w=b.attn.proj.weight, o_b=b.attn.proj.bias, n2w=b.norm2.weight, n2b=b.norm2.bias,
                                  f1_w=b.mlp.fc1.weight, f1_b=b.mlp.fc1.bias, f2_w=b.mlp.fc2.weight,
                                  f2_b=b.mlp.fc2.bias).items():
                setattr(m, name, _P(src))
            self.v_blocks.append(m)
        self.v_nw, self.v_nb, self.v_proj = _P(tr.norm.weight), _P(tr.norm.bias), _P(v.head.proj.weight)
        self.v_heads = tr.blocks[0].attn.num_heads
        bert = t.transformer
        e = bert.embeddings
        self.t_word, self.t_pos = _P(e.word_embeddings.weight), _P(e.position_embeddings.weight)
        self.t_type = _P(e.token_type_embeddings.weight[0])
        self.t_lnw, self.t_lnb, self.t_eps = _P(e.LayerNorm.weight), _P(e.LayerNorm.bias), e.LayerNorm.eps
        self.t_blocks = nn.ModuleList()
        for L in bert.encoder.layer:
            a = L.attention
            m = nn.Module()
            for name, src in dict(
                    qkv_w=torch.cat([a.self.query.weight, a.self.key.weight, a.self.value.weight]),
                    qkv_b=torch.cat([a.self.query.bias, a.self.key.bias, a.self.value.bias]),
                    o_w=a.output.dense.weight, o_b=a.output.dense.bias,
                    n1w=a.output.LayerNorm.weight, n1b=a.output.LayerNorm.bias,
                    f1_w=L.intermediate.dense.weight, f1_b=L.intermediate.dense.bias,
                    f2_w=L.output.dense.weight, f2_b=L.output.dense.bias,
                    n2w=L.output.LayerNorm.weight, n2b=L.output.LayerNorm.bias).items():
                setattr(m, name, _P(src))
            self.t_blocks.append(m)
        self.t_heads = bert.config.num_attention_heads
        self.t_p1, self.t_p2 = _P(t.proj[0].weight), _P(t.proj[2].weight)
        self.logit_scale = _P(ref.logit_scale)
        self.pad_id = t.config.pad_token_id or 0

    # ---- towers ----
    def encode_image(self, x):
        B = x.shape[0]
        H = self.v_heads
        x = F.conv2d(x, self.v_patch_w, self.v_patch_b, stride=16).flatten(2).transpose(1, 2)
        x = torch.cat([self.v_cls.expand(B, -1, -1), x], 1) + self.v_pos
        N, C = x.shape[1], x.shape[2]
        nb = len(self.v_blocks)
        for i, p in enumerate(self.v_blocks):
            h = F.layer_norm(x, (C,), p.n1w, p.n1b, 1e-6)
            qkv = F.linear(h, p.qkv_w, p.qkv_b).view(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            if i == nb - 1:
                q, x = q[:, :, :1], x[:, :1]
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, q.shape[2], C)
            x = x + F.linear(a, p.o_w, p.o_b)
            h = F.layer_norm(x, (C,), p.n2w, p.n2b, 1e-6)
            x = x + F.linear(F.gelu(F.linear(h, p.f1_w, p.f1_b)), p.f2_w, p.f2_b)
        x = F.layer_norm(x[:, 0], (C,), self.v_nw, self.v_nb, 1e-6)
        return F.linear(x, self.v_proj)

    def _block_tail(self, x, a, p, C):
        x = F.layer_norm(x + F.linear(a, p.o_w, p.o_b), (C,), p.n1w, p.n1b, self.t_eps)
        m = F.linear(F.gelu(F.linear(x, p.f1_w, p.f1_b)), p.f2_w, p.f2_b)
        return F.layer_norm(x + m, (C,), p.n2w, p.n2b, self.t_eps)

    def encode_text(self, ids):
        """Padded path: ids [B,L] (L already trimmed to the batch's bucket), key-padding mask when needed."""
        B, L = ids.shape
        H = self.t_heads
        mask = ids != self.pad_id
        am = None if bool(mask.all()) else mask[:, None, None, :]
        x = self.t_word[ids] + self.t_pos[:L] + self.t_type
        C = x.shape[-1]
        x = F.layer_norm(x, (C,), self.t_lnw, self.t_lnb, self.t_eps)
        nb = len(self.t_blocks)
        for i, p in enumerate(self.t_blocks):
            qkv = F.linear(x, p.qkv_w, p.qkv_b).view(B, L, 3, H, C // H).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            if i == nb - 1:
                q, x = q[:, :, :1], x[:, :1]
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=am).transpose(1, 2).reshape(B, q.shape[2], C)
            x = self._block_tail(x, a, p, C)
        x = x[:, 0]
        return F.linear(F.gelu(F.linear(x, self.t_p1)), self.t_p2)

    @staticmethod
    def pack(ids, pad_id=0):
        """CPU-side packing metadata for encode_text_packed. Returns tensors (on ids' device) + max_len int."""
        ids = ids.cpu()
        mask = ids != pad_id
        lens = mask.sum(1)
        tok = ids[mask]
        pos = torch.arange(ids.shape[1]).expand_as(ids)[mask]
        cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(lens, 0)
        return tok, pos, cu, cu[:-1].long(), torch.arange(len(lens) + 1, dtype=torch.int32), int(lens.max())

    def encode_text_packed(self, tok, pos, cu, cls_idx, cu_cls, max_len):
        H = self.t_heads
        x = self.t_word[tok] + self.t_pos[pos] + self.t_type
        C = x.shape[-1]
        D = C // H
        x = F.layer_norm(x, (C,), self.t_lnw, self.t_lnb, self.t_eps)
        nb = len(self.t_blocks)
        for i, p in enumerate(self.t_blocks):
            qkv = F.linear(x, p.qkv_w, p.qkv_b).view(-1, 3, H, D)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            if i == nb - 1:
                q, x = q[cls_idx], x[cls_idx]
                a = torch.ops.aten._flash_attention_forward(q.contiguous(), k, v, cu_cls, cu, 1, max_len, 0.0,
                                                            False, False)[0]
            else:
                a = torch.ops.aten._flash_attention_forward(q, k, v, cu, cu, max_len, max_len, 0.0, False, False)[0]
            x = self._block_tail(x, a.reshape(-1, C), p, C)
        return F.linear(F.gelu(F.linear(x, self.t_p1)), self.t_p2)

    def forward(self, images, ids=None, packed=None):
        fi = self.encode_image(images)
        ft = self.encode_text_packed(*packed) if packed is not None else self.encode_text(ids)
        return fi, ft, self.logit_scale
