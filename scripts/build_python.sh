#!/bin/bash
# Build the native Python extension (PyO3, abi3: any CPython >= 3.9) and place it next to python/clipfast_rs_torch.py.
# Needs nvcc (CUDA >= 12.8 for sm_120a), a Rust toolchain, and optionally CUTLASS headers (CUTLASS_DIR).
set -euo pipefail
cd "$(dirname "$0")/../rs"
CLIPFAST_CUDA_ARCH=${CLIPFAST_CUDA_ARCH:-120a} cargo build --release --features python --lib
T=${CARGO_TARGET_DIR:-target}
cp "$T/release/libclipfast_rs.so" ../python/clipfast_rs.abi3.so
echo "built python/clipfast_rs.abi3.so  (import clipfast_rs; see python/clipfast_rs_torch.py)"
