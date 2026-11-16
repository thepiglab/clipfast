// Opt-in MXFP8 linears (cuBLASLt block scaling: E4M3 values + one UE8M0 power-of-two scale per 32 K-elements).
// Weights are quantized once at load; activations per GEMM. The GEMM writes bf16 to a temp buffer, then one fused
// epilogue kernel applies bias (+ exact GELU) or bias + residual. Attention, LayerNorm and the residual stream stay
// bf16. Scale layout and rounding follow cuBLASLt's "1D Block Scaling Factors Layout" and its "Computing scaling and
// conversion factors" steps (the quantizer is adapted from flashembed, MIT, https://github.com/thepiglab/flashembed).
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace {
// offset of the scale for (row `outer`, K-block `kb`) in the tiled layout: 128 x 4 tiles of 512 bytes
__device__ __forceinline__ size_t sf_offset(int outer, int kb, int sfk_pad) {
  return ((size_t)(outer / 128) * (sfk_pad / 4) + kb / 4) * 512 + (outer % 32) * 16 + ((outer % 128) / 32) * 4 + (kb % 4);
}
// x: bf16 rows (row stride lda) -> q: fp8 [rows, K] contiguous, sf: UE8M0 scales (tiled)
__global__ void k_quant_mx(const __nv_bfloat16* __restrict__ x, long lda, int rows, int K, uint8_t* __restrict__ q,
                           uint8_t* __restrict__ sf, int sfk_pad) {
  const int nb = K / 32;
  const long long total = (long long)rows * nb;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
    const int r = (int)(i / nb), b = (int)(i % nb);
    const __nv_bfloat16* p = x + (size_t)r * lda + (size_t)b * 32;
    float v[32];
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 32; j += 8) {
      const uint4 u = *(const uint4*)(p + j);
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        v[j + 2 * t] = __low2float(h2[t]);
        v[j + 2 * t + 1] = __high2float(h2[t]);
      }
    }
#pragma unroll
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(v[j]));
    int e = 0;
    const float m = frexpf(amax / 448.f, &e);              // S = m * 2^e, m in [0.5, 1)
    int E = amax > 0.f ? (m == 0.5f ? e - 1 : e) : 0;      // round the exponent up unless S is a power of two
    E = max(-127, min(127, E));
    const float R = exp2f((float)-E);
    uint8_t* o = q + (size_t)r * K + (size_t)b * 32;
#pragma unroll
    for (int j = 0; j < 32; j += 16) {
      uint8_t tmp[16];
#pragma unroll
      for (int t = 0; t < 16; ++t) tmp[t] = __nv_cvt_float_to_fp8(v[j + t] * R, __NV_SATFINITE, __NV_E4M3);
      *(uint4*)(o + j) = *(const uint4*)tmp;
    }
    sf[sf_offset(r, b, sfk_pad)] = (uint8_t)(E + 127);
  }
}
// epilogue on the bf16 GEMM result t[M, N]: mode 0 D = t + b; 1 D = gelu(t + b) (exact erf); 2 D += t + b
__global__ void k_epi(const __nv_bfloat16* __restrict__ t, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ D,
                      long ldd, int M, int N, int mode) {
  const long long total = (long long)M * N;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
    const int r = (int)(i / N), c = (int)(i % N);
    float v = __bfloat162float(t[i]) + (bias ? __bfloat162float(bias[c]) : 0.f);
    __nv_bfloat16* d = D + (size_t)r * ldd + c;
    if (mode == 1) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
    else if (mode == 2) v += __bfloat162float(*d);
    *d = __float2bfloat16(v);
  }
}
inline int sfk_pad(int K) { return (K / 32 + 3) / 4 * 4; }
inline unsigned nblk(long long n) { return (unsigned)std::min<long long>((n + 255) / 256, 65535LL * 8); }
}  // namespace

extern "C" {

size_t mx_sf_bytes(int rows, int K) { return (size_t)((rows + 127) / 128) * 128 * sfk_pad(K); }

// quantize rows x K bf16 (row stride lda) into q (rows*K bytes) + sf (mx_sf_bytes) on stream st
int mx_quantize(const void* x, long lda, int rows, int K, void* q, void* sf, cudaStream_t st) {
  if (K % 32) return -2;
  const size_t full = (size_t)(rows / 128) * (sfk_pad(K) / 4) * 512, all = mx_sf_bytes(rows, K);
  if (all > full) cudaMemsetAsync((uint8_t*)sf + full, 0, all - full, st);  // tiles past the last row: zero scales
  k_quant_mx<<<nblk((long long)rows * (K / 32)), 256, 0, st>>>((const __nv_bfloat16*)x, lda, rows, K, (uint8_t*)q,
                                                               (uint8_t*)sf, sfk_pad(K));
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// D[M,N] (row stride ldd) (=|+=) epi(X[M,K] (bf16, row stride lda) * W[N,K]^T + bias), W pre-quantized (wq, wsf).
// qx/qsf: activation quantization buffers; tmp: bf16 [M, N] contiguous. mode: 0 bias, 1 bias+gelu, 2 bias+residual.
int mx_gemm(cublasLtHandle_t lt, void* D, long ldd, const void* X, long lda, int M, const void* wq, const void* wsf,
            int N, int K, const void* bias, int mode, void* qx, void* qsf, void* tmp, void* ws, size_t ws_bytes,
            cudaStream_t st) {
  if (mx_quantize(X, lda, M, K, qx, qsf, st)) return -1;
  cublasLtMatmulDesc_t d;
  if (cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return -1;
  const cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
  const int32_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
  cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
  cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
  cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
  cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &wsf, sizeof(wsf));
  cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &qsf, sizeof(qsf));
  cublasLtMatrixLayout_t la, lb, lc;
  cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, K, N, K);
  cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, K, M, K);
  cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N);
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes));
  cublasLtMatmulHeuristicResult_t res;
  int got = 0;
  cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(lt, d, la, lb, lc, lc, pref, 1, &res, &got);
  cublasLtMatmulPreferenceDestroy(pref);
  int rc = 0;
  if (s != CUBLAS_STATUS_SUCCESS || got < 1) {
    fprintf(stderr, "cuBLASLt has no MXFP8 kernel for N%d K%d M%d\n", N, K, M);
    rc = -3;
  } else {
    const float al = 1.f, be = 0.f;
    if (cublasLtMatmul(lt, d, &al, wq, la, qx, lb, &be, tmp, lc, tmp, lc, &res.algo, ws, ws_bytes, st) != CUBLAS_STATUS_SUCCESS)
      rc = -4;
  }
  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulDescDestroy(d);
  if (rc) return rc;
  k_epi<<<nblk((long long)M * N), 256, 0, st>>>((const __nv_bfloat16*)tmp, (const __nv_bfloat16*)bias,
                                               (__nv_bfloat16*)D, ldd, M, N, mode);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
}
