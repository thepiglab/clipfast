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
  if (variant == 1) return run<GeluGemm<128, 64, 4>>(D, A, lda, W, bias, M, N, K, st);
  return run<GeluGemm<256, 64, 3>>(D, A, lda, W, bias, M, N, K, st);
}
#else
extern "C" int cf_gemm_bias_gelu_bf16(void*, const void*, long, const void*, const void*, int, int, int, cudaStream_t,
                                      int) {
  return -1;
}
#endif
