// Exact (erf) GELU fused into the fc1 GEMM epilogue: D = GELU(A @ W^T + bias), bf16 in/out, fp32 accumulate and
// epilogue, one bf16 rounding. CUTLASS 2.x device GEMM with the same 128x256x32 sm80 tensor-op tile cuBLAS selects
// for these shapes on RTX PRO 6000; the bias is fed as the C operand with leading dimension 0 (row broadcast).
// Built only when CUTLASS headers are available (CLIPFAST_CUTLASS=1 from build.rs); otherwise returns -1.
#include <cuda_runtime.h>

#ifdef CLIPFAST_CUTLASS
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"
#include "cutlass/gemm/device/gemm.h"

namespace {
using bf16 = cutlass::bfloat16_t;
template <int BN, int WN, int STAGES>
using GeluGemm = cutlass::gemm::device::Gemm<
    bf16, cutlass::layout::RowMajor,        // A [M, K]
    bf16, cutlass::layout::ColumnMajor,     // B = W^T: W [N, K] row-major == [K, N] column-major
    bf16, cutlass::layout::RowMajor,        // C (bias, ldc = 0) and D [M, N]
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, BN, 32>, cutlass::gemm::GemmShape<64, WN, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationGELU<bf16, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, STAGES>;

// plain bias GEMM (D = A W^T + b; bias as C with ldc = 0), for the qkv projection
template <int BN, int STAGES, int SW>
using BiasGemm = cutlass::gemm::device::Gemm<
    bf16, cutlass::layout::RowMajor, bf16, cutlass::layout::ColumnMajor, bf16, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, BN, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<bf16, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SW>, STAGES>;

template <typename G>
int run(void* D, const void* A, long lda, const void* W, const void* bias, int M, int N, int K, cudaStream_t st) {
  typename G::Arguments args({M, N, K}, {(const bf16*)A, (int)lda}, {(const bf16*)W, K}, {(const bf16*)bias, 0},
                             {(bf16*)D, N}, {1.f, 1.f});
  G op;
  if (op.can_implement(args) != cutlass::Status::kSuccess) return -2;
  if (op.initialize(args, nullptr, st) != cutlass::Status::kSuccess) return -3;
  return op(st) == cutlass::Status::kSuccess ? 0 : -4;
}
}  // namespace

extern "C" int cf_gemm_bias_gelu_bf16(void* D, const void* A, long lda, const void* W, const void* bias, int M, int N,
                                      int K, cudaStream_t st, int variant) {
  // tiles from the kar vit_gemm sweep (research/kar_tasks/vit_gemm): 0 = 128x256 s3 (cuBLASLt's pick),
  // 1 = 128x128 s4 (best for M ~ 2-16k), 2 = 128x256 s4 (best for M >= 16k)
  if (variant == 1) return run<GeluGemm<128, 64, 4>>(D, A, lda, W, bias, M, N, K, st);
  if (variant == 2) return run<GeluGemm<256, 64, 4>>(D, A, lda, W, bias, M, N, K, st);
  return run<GeluGemm<256, 64, 3>>(D, A, lda, W, bias, M, N, K, st);
}

// D = A W^T + bias. variant 0: 128x256 s3 swizzle 16 (M >= 16k), 1: 128x128 s4 (M ~ 2-16k)
extern "C" int cf_gemm_bias_bf16(void* D, const void* A, long lda, const void* W, const void* bias, int M, int N,
                                 int K, cudaStream_t st, int variant) {
  if (variant == 1) return run<BiasGemm<128, 4, 8>>(D, A, lda, W, bias, M, N, K, st);
  return run<BiasGemm<256, 3, 16>>(D, A, lda, W, bias, M, N, K, st);
}
#else
extern "C" int cf_gemm_bias_gelu_bf16(void*, const void*, long, const void*, const void*, int, int, int, cudaStream_t,
                                      int) {
  return -1;
}
extern "C" int cf_gemm_bias_bf16(void*, const void*, long, const void*, const void*, int, int, int, cudaStream_t, int) {
  return -1;
}
#endif
