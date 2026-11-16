# clipfast

Fast, parity-first inference for **BiomedCLIP** (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`:
ViT-B/16 image tower + PubMedBERT text tower) on NVIDIA Blackwell RTX PRO 6000 (sm_120).

The main interface is **`clipfast-rs`**, a standalone Rust + CUDA runtime: cuBLASLt and CUTLASS GEMMs plus custom
kernels, safetensors weights, a fixed preallocated GPU workspace and one CUDA graph per shape bucket. No Python,
no PyTorch and no JIT compile at run time.
An **optional Python binding** exposes the same engine to PyTorch code (see below).

## Results (RTX PRO 6000 Blackwell Server Edition 96 GB, CUDA 13, one GPU)

Baselines: open_clip in fp32, open_clip with bf16 autocast ("AMP", what most people run), and open_clip with bf16
weights + `torch.compile(mode="max-autotune")` + CUDA graphs (the strongest PyTorch baseline we found).

### Parity vs open_clip fp32 (TF32 off)

| | image cos (min) | text cos (min) | max logit diff | zero-shot top-1 agreement |
|---|---|---|---|---|
| open_clip AMP | 0.99993 | 0.99999 | 0.15 | 9/9 |
| open_clip torch.compile | 0.99951 | 0.99922 | 0.26 | 9/9 |
| **clipfast-rs bf16 (default)** | **0.99991** | **0.99993** | 0.29 | 9/9 |
| **clipfast-rs fp32 (parity mode)** | **0.9999999** | **0.9999999** | **3e-5** | 9/9 |

Probe set: the model's example images + synthetic images, 58 medical prompts; image->text and text->image top-1
retrieval agree with the reference except on near-tied synthetic images.

### Speed (GPU forward, inputs resident)

| | open_clip AMP | open_clip compile | **clipfast-rs** |
|---|---|---|---|
| image, 1 image | 1.56 ms | 0.65 ms | **0.63 ms** |
| image, batch 1024 | 5.3k img/s | 6.9k img/s | **8.4k img/s** (1.22x compile, 1.59x AMP) |
| text, 1 query (32 tokens) | 2.29 ms | 0.78 ms | **0.50 ms** |
| text, 1024 x 32 tokens | 263 ms | 186 ms | **17.7 ms** (10.5x) |
| text, 1024 x 256 tokens | 263 ms | 186 ms | **146 ms** (1.27x) |
| text, 1024 PubMed-caption-like lengths (median 40) | 271 ms | 185 ms | **30.0 ms** packed (6.2x, 9.0x AMP) |

### GPU memory (whole process)

| | weights | device memory in use |
|---|---|---|
| open_clip AMP | 747 MB (fp32) | 3.3-11.3 GB |
| open_clip compile | 419 MB (bf16) | 11.8-14.4 GB |
| **clipfast-rs bf16** | **374 MB** | **2.3 GB, flat at every batch size** |

Weights load in ~0.25 s; full speed after one warm-up call per shape bucket.

## Why it is faster

| change | effect |
|---|---|
| no padding to 256 tokens; CLS-only last layer in both towers (the embedding only reads CLS) | up to 10x on short captions, exact |
| packed variable-length text (only real tokens computed) + flash-style attention over sequence offsets | 4.3x on caption-like length mixes, exact |
| fused epilogues: residual via cuBLASLt beta*C + bias; fc1 + **exact erf** GELU fused in a CUTLASS GEMM | -11% image time at batch 1024, parity unchanged |
| fixed workspace + CUDA graph per bucket | small-batch latency, flat memory |

Tried and dropped (measured): cuBLASLt heuristic autotune (no gain), cuBLASLt tanh-GELU epilogue (same speed,
lower parity than exact erf), fp8 e4m3 (1.42x at large batch but cos ~0.998, kept as an opt-in experiment in the
PyTorch research path only).

Kernel work followed the kernel-autoresearch method (freeze a task, prove the evaluator with cheating canaries,
calibrate the GPU, A/B with confidence intervals, keep only verified wins); task definitions are in `kar_tasks/`.
One finding: cuBLAS's small-M path reduces split-K partials in bf16, 1.9x the error of fp32 accumulation; our
skinny kernel accumulates in fp32.

## Using it

Everything below is Rust + CUDA; Python appears only in the optional extension and in `research/`.

```bash
# 1. export runtime assets straight from the open_clip checkpoint folder (no PyTorch needed)
clipfast-rs export --model-dir /path/to/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 --out assets

# 2a. embed files: a folder (or list) of JPEG/PNG/... images and/or a file with one caption per line
clipfast-rs embed --assets assets --images ./figures --texts captions.txt --out run1
#    -> run1.images.npy / run1.images.txt (paths), run1.texts.npy   (float32 [N, 512], L2-normalized)

# 2b. or serve it over HTTP
clipfast-rs serve --assets assets --port 8080
curl -s localhost:8080/v1/embed/text -d '{"texts": ["chest x-ray with bilateral opacities"]}'
curl -s localhost:8080/v1/embed/image -H 'Content-Type: image/jpeg' --data-binary @figure.jpg
```

Front-end parity, measured against open_clip's Python path:
- **Pixels:** bit-identical (max diff 0.0) on 47 files: JPEG 4:4:4 and 4:2:0, grayscale JPEG, PNG RGB, RGBA and
  palette, sizes from 97x333 to 3000x2400, and the model's example images.
- **Token ids:** identical on probe prompts plus unicode, markup and over-length text.
- **`embed` end to end:** cos 0.99986 (bf16), 0.9999998 (fp32), with top-1 retrieval identical.
- **Exported weights:** byte-identical to a PyTorch-side export (302 tensors, f32 and bf16).

What the front end does, all in Rust:
- **Tokenizer:** the `tokenizers` crate reading the model's own `tokenizer.json`, with open_clip's whitespace
  cleanup. Captions are truncated to 256 tokens and packed (no padding).
- **Images:** decoded with libjpeg-turbo (JPEG, the same decoder Pillow uses) or the `image` crate (PNG, BMP, TIFF,
  WebP, GIF). They are then resized with a re-implementation of Pillow's fixed-point bicubic resampler and
  center-cropped, matching open_clip's `Image.open(p).convert("RGB")` + eval transform. Decoding runs on a CPU thread
  pool and overlaps GPU encoding.
- **Weights:** the PyTorch checkpoint (zip + pickle) is read directly, and the q/k/v fusion and folding are done in
  Rust.

Other commands: `parity` (against golden embeddings from `research/golden.py`), `bench`, `prep` (dump
preprocessed pixels) and `tokenize`.

## Retrieval: int8 index with exact re-rank

`clipfast-rs search` scores the whole corpus in int8 and then re-ranks the top 200 per query in exact fp32:
- **Corpus:** int8 with one global scale, taken from the 99.9th percentile of per-vector max |x|.
- **Queries:** int8 with a per-query scale.
- **Scoring:** a `dp4a` kernel for 8 or fewer queries, and a CUTLASS int8 tensor-core GEMM above that.
- **Re-rank copy:** the fp32 copy lives on the GPU ("device"), or in pinned host memory ("host": only about 200
  rows per query cross PCIe).

Measured on 2M x 512 vectors (200k real BiomedCLIP text embeddings plus 1.8M clustered synthetic vectors), top-10:

| queries | exact fp32 | int8 + exact re-rank | top-10 identical to exact |
|---|---|---|---|
| 1 | 3.58 ms | **1.69 ms** | 1/1 |
| 8 | 3.99 ms | **2.22 ms** | 8/8 |
| 256 | 19.5 ms | **11.7 ms** | 256/256 |
| 2048 | 130 ms | **92.4 ms** | 2045/2048 (the 3 rows differ only by ties within 1.1e-6) |

GPU exact search and int8 + re-rank both match a CPU brute-force reference.

Memory per million 512-d vectors:

| re-rank copy | GPU memory | pinned host memory | speed (1 / 256 / 2048 queries) |
|---|---|---|---|
| device | 2.44 GB | none | as in the table above |
| host | 0.49 GB | 1.95 GB | 1.73 ms / 15.0 ms / 109 ms |

```bash
clipfast-rs search --corpus run1.images.npy --names run1.images.txt --assets assets --texts queries.txt --k 10 [--rerank host]
clipfast-rs search-bench --assets assets --real 200000 --synth 1800000 --nq 1,8,256,2048
```

## Lower-precision modes we measured (and do not recommend on RTX PRO 6000)

| mode | parity vs open_clip fp32 | speed vs bf16 | verdict |
|---|---|---|---|
| `--quant mxfp8` (cuBLASLt block-scaled E4M3, scale per 32 values) | cos 0.9973 (image) / 0.9982 (text) | 0.76x (6.3k vs 8.3k img/s) | slower and less exact: the activation-quantize pass and the unfused epilogue cost more than the FP8 GEMM saves at ViT-B/BERT-base sizes |
| `CLIPFAST_FP32_EMU=1` (BF16x9-emulated fp32) | identical to fp32 | 1.00x | no gain: native fp32 (82 TFLOPS here) beats bf16x9 (437/9) and 3xTF32 (217/3); worthwhile on datacenter GPUs only |


## Build

```bash
cd rs
CUDA_HOME=/usr/local/cuda CUTLASS_DIR=/path/to/cutlass CLIPFAST_CUDA_ARCH=120a cargo build --release
# -> target/release/clipfast-rs
```

Requirements: CUDA >= 12.8 for sm_120a (`CLIPFAST_CUDA_ARCH=100a,120a` also targets B200). CUTLASS headers
(v3.9.x, not vendored) enable the fused exact-GELU GEMM. Build on local disk, because rust-lld can fail on some
network filesystems.

## Python extension (optional)

A native extension (PyO3, abi3, any CPython >= 3.9) exposes the same engine to PyTorch code, for example to
embed a validation set during training. Tokenization, shape bucketing and text packing run in Rust; tensors are
passed by address, on CUDA or CPU.

```bash
scripts/build_python.sh          # -> python/clipfast_rs.abi3.so
```

```python
from clipfast_rs_torch import ClipFast            # python/clipfast_rs_torch.py, ~30 lines over `import clipfast_rs`
m = ClipFast("assets")                            # fp32=True for parity mode
img = m.encode_images(pixels)                     # float [B,3,224,224] open_clip-preprocessed -> cuda [B,512]
txt = m.encode_texts(["chest x-ray with bilateral opacities"])
logits = m.logit_scale * img @ txt.T
```

From Python, the extension matches the CLI: image batch 1024 runs at 8.40k img/s (CLI 8.43k). 1,024 captions take
48 ms from raw strings including tokenization, and 35 ms from token ids (CLI 30 ms; the gap is converting Python
lists to Rust vectors). Parity is identical (cos 0.99990 / 0.99992 bf16, 0.9999999 fp32; token ids match open_clip).

## Repository layout

| path | what |
|---|---|
| `rs/src/engine.rs`, `rs/cuda/` | GPU engine: CUDA kernels (`clip.cu`), CUTLASS fused GELU GEMM (`gemm_gelu.cu`), batching |
| `rs/src/frontend.rs`, `rs/src/prep.rs` | tokenizer, image decode + open_clip-identical preprocessing, `.npy` output |
| `rs/src/export.rs` | PyTorch checkpoint reader + runtime safetensors writer |
| `rs/src/server.rs` | HTTP server |
| `rs/src/search.rs`, `rs/cuda/search.cu`, `rs/cuda/cutlass_int8.cu` | int8 index + exact re-rank |
| `rs/cuda/mxfp8.cu` | opt-in MXFP8 linears |
| `rs/src/main.rs` | CLI (`export`, `embed`, `serve`, `search`, `search-bench`, `parity`, `bench`, `prep`, `tokenize`) |
| `rs/src/py.rs`, `python/` | optional native Python extension + torch wrapper |
| `research/` | PyTorch reference runtime and benchmarks against open_clip, parity goldens and tests, data-feeding benchmarks, kernel-autoresearch tasks |

## Feeding the GPU

On Zen 5 EPYC, PIL decode + open_clip preprocess runs at ~335 img/s per core for 512 px JPEGs and ~37 for
2048 px, far below the ~8k img/s the GPU consumes. Decode once into a uint8 224 px cache, or decode on the GPU
(nvJPEG decoded 24.6k img/s at 512 px on one GPU). If you use torchvision's batched `decode_jpeg(device="cuda")`,
decode on a side stream and synchronize it before use; we saw a race that produced NaN images otherwise.

## Credits

BiomedCLIP (Microsoft, MIT), open_clip, NVIDIA CUTLASS (BSD-3, fetched at build time, not vendored). The int8
search kernels and the MXFP8 quantizer are adapted from [flashcluster](https://github.com/thepiglab/flashcluster)
and [flashembed](https://github.com/thepiglab/flashembed) (MIT). The kernel-autoresearch method (https://github.com/gauravjain14/kernel-autoresearch-plugin, Apache-2.0).
