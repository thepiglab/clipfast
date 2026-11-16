"""Parity goldens for clipfast-rs (needs PyTorch + open_clip, since the reference IS open_clip; CPU only):
  golden.safetensors - probe pixels (float32 NCHW, open_clip preprocess), probe token ids (int32, padded to 256),
      open_clip fp32 reference embeddings (normalized), for `clipfast-rs parity`.
  golden_texts.json  - the probe prompts.
Weights/config/tokenizer come from the Rust exporter: `clipfast-rs export --model-dir DIR --out OUT_DIR`.
python research/golden.py OUT_DIR
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
    print("golden written:", px.shape, ids.shape, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
