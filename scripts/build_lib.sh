#!/bin/bash
# Build the CUDA engine as a shared library for the optional Python binding (clipfast/rs_py.py).
# Needs nvcc (CUDA >= 12.8 for sm_120a) and CUTLASS headers (CUTLASS_DIR; v3.9.x tested) for the fused erf-GELU GEMM.
# usage: CUTLASS_DIR=/path/to/cutlass scripts/build_lib.sh [sm-arch, default 120a]
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH=${1:-120a}
mkdir -p build
DEFS=()
[ -n "${CUTLASS_DIR:-}" ] && DEFS=(-I"$CUTLASS_DIR/include" -DCLIPFAST_CUTLASS=1)
nvcc -shared -Xcompiler -fPIC -O3 -std=c++17 --expt-relaxed-constexpr -gencode=arch=compute_$ARCH,code=sm_$ARCH \
  "${DEFS[@]}" rs/cuda/clip.cu rs/cuda/gemm_gelu.cu -cudart static -lcublasLt -o build/libclipfast_cuda.so
echo "built build/libclipfast_cuda.so"
