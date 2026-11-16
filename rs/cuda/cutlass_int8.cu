// CUTLASS 2.x int8 tensor-core GEMM (mma.sync m16n8k32, Ampere-style; runs on SM80+ incl. SM120 / RTX Blackwell)
// with an f16 output epilogue D = alpha * acc. Layouts: A corpus chunk row-major m x k, B queries column-major k x n,
// D column-major m x n (= row-major [query][corpus row], ld = m). Adapted from flashcluster (MIT,
// https://github.com/thepiglab/flashcluster). Without CUTLASS headers the entry point returns -1 (cuBLAS fallback).
#include <cuda_runtime.h>

#ifdef CLIPFAST_CUTLASS
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"

namespace {
using Epi = cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, int32_t, float,
                                                         cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
using Gemm80 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor, cutlass::half_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
    Epi, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
}  // namespace

extern "C" int cs_cutlass_int8_gemm(const void* A, const void* B, void* D, int m, int n, int k, float alpha, cudaStream_t st) {
  Gemm80 g;
  Gemm80::Arguments args({m, n, k}, {(const int8_t*)A, k}, {(const int8_t*)B, k}, {(cutlass::half_t*)D, m},
                         {(cutlass::half_t*)D, m}, {alpha});
  if (g.can_implement(args) != cutlass::Status::kSuccess) return -2;
  if (g.initialize(args, nullptr, st) != cutlass::Status::kSuccess) return -3;
  return g.run(st) == cutlass::Status::kSuccess ? 0 : -4;
}
#else
extern "C" int cs_cutlass_int8_gemm(const void*, const void*, void*, int, int, int, float, cudaStream_t) { return -1; }
#endif
