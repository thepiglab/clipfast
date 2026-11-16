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
PyTorch path only).

Kernel work followed the kernel-autoresearch method (freeze a task, prove the evaluator with cheating canaries,
calibrate the GPU, A/B with confidence intervals, keep only verified wins); task definitions are in `kar_tasks/`.
One finding: cuBLAS's small-M path reduces split-K partials in bf16, 1.9x the error of fp32 accumulation; our
skinny kernel accumulates in fp32.

## Status

- `clipfast-rs` runs the full model from exported tensors (pixels already preprocessed, token ids):
  `parity` and `bench` commands. **Next for production use:** Rust tokenizer (`tokenizers` crate on the model's own
  `tokenizer.json`), open_clip-identical JPEG/PNG decode + resize/crop on CPU threads (optional nvJPEG), and an
  `embed` command / HTTP server.
- Training is out of scope here (PyTorch; a lean model reference is in `clipfast/train_model.py`).

## Build and run (Rust)

```bash
# 1. export weights + parity goldens from a local copy of the model (Python, CPU; needs open_clip, safetensors)
BIOMEDCLIP_DIR=/path/to/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 python export_rs.py rs_assets

# 2. build (CUDA >= 12.8 for sm_120a; CUTLASS headers enable the fused erf-GELU GEMM)
cd rs
CUDA_HOME=/usr/local/cuda CUTLASS_DIR=/path/to/cutlass CLIPFAST_CUDA_ARCH=120a cargo build --release

# 3. run
./target/release/clipfast-rs parity --assets ../rs_assets            # add --fp32 for parity mode, --packed for varlen text
./target/release/clipfast-rs bench  --assets ../rs_assets --img-b 1,64,1024 --txt-b 1,1024 --txt-l 32,256,pmc --packed
```

Build on local disk (rust-lld can fail on some network filesystems). `CLIPFAST_CUDA_ARCH=100a,120a` also
targets B200.

## Optional: Python binding

`clipfast/rs_py.py` loads the same CUDA engine as a shared library (ctypes) and passes torch tensors by pointer,
for use inside PyTorch code, e.g. embedding a validation set during training. It is a convenience layer, not the main
interface: it is **~7% slower than the Rust CLI** in places because shape bucketing and text packing run in Python
(image batch 1024: 7.85k vs 8.4k img/s; 1024 caption-length texts: 36.8 vs 30.0 ms; 79 ms from raw strings
including tokenization). Parity is identical (tokenizer ids match open_clip; same embeddings).

```bash
CUTLASS_DIR=/path/to/cutlass scripts/build_lib.sh          # -> build/libclipfast_cuda.so
```

```python
from clipfast.rs_py import ClipFastRS
m = ClipFastRS("rs_assets")                 # fp32=True for parity mode
img = m.encode_images(pixels)               # float [B,3,224,224], open_clip-preprocessed -> cuda [B,512], L2-normalized
txt = m.encode_texts(["chest x-ray with bilateral opacities", ...])
logits = m.logit_scale * img @ txt.T
```

## Repository layout

| path | what |
|---|---|
| `rs/` | Rust crate (`src/main.rs`) + CUDA engine (`cuda/clip.cu`, `cuda/gemm_gelu.cu`) |
| `clipfast/rs_py.py` | optional Python binding |
| `clipfast/fast.py`, `clipfast/kernels.py` | PyTorch research runtime (CUDA graphs, Triton fused epilogues, skinny GEMM, fp8 experiment) |
| `export_rs.py` | weights + golden parity export for the Rust runtime |
| `bench.py`, `report.py` | PyTorch benchmark matrix (parity, speed, memory) vs open_clip baselines |
| `cpu_feed_bench.py`, `gpu_feed_bench.py` | data-feeding throughput (PIL / torchvision / nvJPEG) |
| `kar_tasks/` | frozen kernel tasks (skinny linear, fused-epilogue linear) |
| `clipfast/train_model.py`, `train_bench.py` | lean trainable model + training-step benchmark (reference) |

## Feeding the GPU

On Zen 5 EPYC, PIL decode + open_clip preprocess runs at ~335 img/s per core for 512 px JPEGs and ~37 for
2048 px, far below the ~8k img/s the GPU consumes. Decode once into a uint8 224 px cache, or decode on the GPU
(nvJPEG decoded 24.6k img/s at 512 px on one GPU). If you use torchvision's batched `decode_jpeg(device="cuda")`,
decode on a side stream and synchronize it before use; we saw a race that produced NaN images otherwise.

## Credits

BiomedCLIP (Microsoft, MIT), open_clip, NVIDIA CUTLASS (BSD-3, fetched at build time, not vendored), and the
kernel-autoresearch method (https://github.com/gauravjain14/kernel-autoresearch-plugin, Apache-2.0).
