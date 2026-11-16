"""Export BiomedCLIP for clipfast-rs (CPU only):
  weights_bf16.safetensors / weights_f32.safetensors  - flat tensors with stable names (q/k/v fused for BERT,
      patch conv as a [768, 768] GEMM weight in (c, ky, kx) order, cls+pos0 folded, token-type 0 folded)
  golden.safetensors - probe pixels (float32 NCHW, open_clip preprocess), probe token ids (int32, padded to 256),
      open_clip fp32 reference embeddings (normalized) and logit scale, for parity tests.
  config.json - dims, eps, pad id, mean/std.
python export_rs.py OUT_DIR
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clipfast.load import load_reference, probe_images, probe_texts  # noqa: E402


def main(out):
    os.makedirs(out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    ref, preprocess, tok = load_reference("cpu")
    ref.eval()
    tr, t = ref.visual.trunk, ref.text
    W = {}
    W["v.patch.w"] = tr.patch_embed.proj.weight.reshape(768, -1)          # [768, 3*16*16], (c, ky, kx)
    W["v.patch.b"] = tr.patch_embed.proj.bias
    W["v.cls_pos0"] = tr.cls_token[0, 0] + tr.pos_embed[0, 0]
    W["v.pos"] = tr.pos_embed[0, 1:]
    for i, b in enumerate(tr.blocks):
        p = f"v.{i}."
        W[p + "ln1.w"], W[p + "ln1.b"] = b.norm1.weight, b.norm1.bias
        W[p + "qkv.w"], W[p + "qkv.b"] = b.attn.qkv.weight, b.attn.qkv.bias
        W[p + "o.w"], W[p + "o.b"] = b.attn.proj.weight, b.attn.proj.bias
        W[p + "ln2.w"], W[p + "ln2.b"] = b.norm2.weight, b.norm2.bias
        W[p + "fc1.w"], W[p + "fc1.b"] = b.mlp.fc1.weight, b.mlp.fc1.bias
        W[p + "fc2.w"], W[p + "fc2.b"] = b.mlp.fc2.weight, b.mlp.fc2.bias
    W["v.ln.w"], W["v.ln.b"] = tr.norm.weight, tr.norm.bias
    W["v.proj.w"] = ref.visual.head.proj.weight
    bert = t.transformer
    e = bert.embeddings
    W["t.word"] = e.word_embeddings.weight
    W["t.pos_type"] = e.position_embeddings.weight + e.token_type_embeddings.weight[0]
    W["t.ln.w"], W["t.ln.b"] = e.LayerNorm.weight, e.LayerNorm.bias
    for i, L in enumerate(bert.encoder.layer):
        a = L.attention
        p = f"t.{i}."
        W[p + "qkv.w"] = torch.cat([a.self.query.weight, a.self.key.weight, a.self.value.weight])
        W[p + "qkv.b"] = torch.cat([a.self.query.bias, a.self.key.bias, a.self.value.bias])
        W[p + "o.w"], W[p + "o.b"] = a.output.dense.weight, a.output.dense.bias
        W[p + "ln1.w"], W[p + "ln1.b"] = a.output.LayerNorm.weight, a.output.LayerNorm.bias
        W[p + "fc1.w"], W[p + "fc1.b"] = L.intermediate.dense.weight, L.intermediate.dense.bias
        W[p + "fc2.w"], W[p + "fc2.b"] = L.output.dense.weight, L.output.dense.bias
        W[p + "ln2.w"], W[p + "ln2.b"] = L.output.LayerNorm.weight, L.output.LayerNorm.bias
    W["t.p1.w"], W["t.p2.w"] = t.proj[0].weight, t.proj[2].weight
    W["logit_scale"] = ref.logit_scale.detach().reshape(1)
    W = {k: v.detach().float().contiguous() for k, v in W.items()}
    save_file(W, os.path.join(out, "weights_f32.safetensors"))
    save_file({k: v.to(torch.bfloat16) for k, v in W.items()}, os.path.join(out, "weights_bf16.safetensors"))

    px, _ = probe_images(preprocess)
    labels = ["adenocarcinoma histopathology", "brain MRI", "covid line chart", "squamous cell carcinoma "
              "histopathology", "immunohistochemistry histopathology", "bone X-ray", "chest X-ray", "pie chart",
              "hematoxylin and eosin histopathology"]
    tmpl = ["this is a photo of {}", "an image of {}", "{}", "a medical image showing {}", "a figure from a paper: {}"]
    texts = probe_texts() + [x.format(l) for x in tmpl for l in labels]
    ids = tok(texts)
    with torch.inference_mode():
        ei = ref.encode_image(px, normalize=True).float()
        et = ref.encode_text(ids, normalize=True).float()
    save_file({"pixels": px.float().contiguous(), "ids": ids.to(torch.int32).contiguous(), "img_emb": ei.contiguous(),
               "txt_emb": et.contiguous()}, os.path.join(out, "golden.safetensors"))
    json.dump({"texts": texts}, open(os.path.join(out, "golden_texts.json"), "w"), indent=1)
    cfg = dict(width=768, heads=12, layers=12, mlp=3072, embed=512, text_proj_hidden=640, img=224, patch=16,
               tokens=197, vocab=int(e.word_embeddings.weight.shape[0]), max_pos=int(e.position_embeddings.weight.shape[0]),
               v_eps=1e-6, t_eps=float(e.LayerNorm.eps), pad_id=int(t.config.pad_token_id or 0),
               cls_id=2, sep_id=3, logit_scale=float(ref.logit_scale.exp()))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    print("exported", len(W), "tensors;", px.shape, ids.shape, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
