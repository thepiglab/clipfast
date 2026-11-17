// kar task vit_gemm: the four ViT-B/16 linear shapes at image batch 256 (M = 256 * 197 = 50432), bf16 in/out,
// fp32 accumulate, bias epilogue. Frozen shapes; reference = cuBLASLt's heuristic pick (what the engine runs today).
// Candidates: CUTLASS 2.x sm80 tensor-op GEMMs (mma.sync m16n8k16, the instruction sm_120 runs) with different
// threadblock tiles / pipeline stages. Each candidate is checked against cuBLASLt's output before it is timed.
//
// nvcc -O3 -std=c++17 -arch=sm_120a --expt-relaxed-constexpr -I$CUTLASS_DIR/include sweep.cu -lcublasLt -o sweep
// ./sweep [M] [reps]
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { printf("cuBLASLt %s: %d\n", #x, (int)s); exit(1); } } while (0)

using bf16 = cutlass::bfloat16_t;
constexpr double PEAK = 437.0;  // calibrated dense bf16 TFLOPS on this GPU (kar calibrate)

template <int TM, int TN, int TK, int WM, int WN, int ST, int SW>
using G = cutlass::gemm::device::Gemm<
    bf16, cutlass::layout::RowMajor, bf16, cutlass::layout::ColumnMajor, bf16, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<TM, TN, TK>,
    cutlass::gemm::GemmShape<WM, WN, TK>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<bf16, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SW>, ST>;

struct Cand {
  std::string name;
  std::function<int(void*, const void*, const void*, const void*, int, int, int, cudaStream_t)> fn;
};

template <typename Op>
Cand make(const char* name) {
  return {name, [](void* D, const void* A, const void* W, const void* b, int M, int N, int K, cudaStream_t st) {
            typename Op::Arguments args({M, N, K}, {(const bf16*)A, K}, {(const bf16*)W, K}, {(const bf16*)b, 0},
                                        {(bf16*)D, N}, {1.f, 1.f});
            Op op;
            if (op.can_implement(args) != cutlass::Status::kSuccess) return -2;
            if (op.initialize(args, nullptr, st) != cutlass::Status::kSuccess) return -3;
            return op(st) == cutlass::Status::kSuccess ? 0 : -4;
          }};
}

__global__ void fill(__nv_bfloat16* p, long n, unsigned seed, float scale) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    unsigned h = (unsigned)i * 2654435761u ^ seed;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    p[i] = __float2bfloat16(((h & 0xffff) / 65535.f - 0.5f) * 2.f * scale);
  }
}

// D = A W^T + b with cuBLASLt (same column-major view as the engine); returns ms per call
struct Lt {
  cublasLtHandle_t h;
  void* ws;
  size_t wsb = 64 << 20;
  Lt() { CB(cublasLtCreate(&h)); CK(cudaMalloc(&ws, wsb)); }
  void run(void* D, const void* A, const void* W, const void* b, int M, int N, int K, cudaStream_t st) {
    cublasLtMatmulDesc_t op;
    CB(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t t = CUBLAS_OP_T, n = CUBLAS_OP_N;
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t)));
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &n, sizeof(n)));
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &b, sizeof(b)));
    cublasLtMatrixLayout_t la, lb, lc;
    CB(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K));
    CB(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, M, K));
    CB(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N));
    cublasLtMatmulPreference_t pref;
    CB(cublasLtMatmulPreferenceCreate(&pref));
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof(wsb)));
    cublasLtMatmulHeuristicResult_t r;
    int got = 0;
    CB(cublasLtMatmulAlgoGetHeuristic(h, op, la, lb, lc, lc, pref, 1, &r, &got));
    const float al = 1.f, be = 0.f;
    CB(cublasLtMatmul(h, op, &al, W, la, A, lb, &be, D, lc, D, lc, &r.algo, ws, wsb, st));
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
  }
};

__global__ void diff(const __nv_bfloat16* a, const __nv_bfloat16* b, long n, float* out) {
  float m = 0.f, s = 0.f;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    float x = __bfloat162float(a[i]), y = __bfloat162float(b[i]);
    m = fmaxf(m, fabsf(x - y));
    s = fmaxf(s, fabsf(y));
  }
  atomicMax((int*)out, __float_as_int(m));
  atomicMax((int*)out + 1, __float_as_int(s));
}

// median-of-reps ms per call, each rep timing `iters` back-to-back launches with events
template <typename F>
std::vector<double> timeit(F f, cudaStream_t st, int reps, int iters) {
  for (int i = 0; i < 3; i++) f();
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  std::vector<double> v;
  for (int r = 0; r < reps; r++) {
    cudaEventRecord(e0, st);
    for (int i = 0; i < iters; i++) f();
    cudaEventRecord(e1, st);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    v.push_back(ms / iters);
  }
  std::sort(v.begin(), v.end());
  return v;
}

int main(int argc, char** argv) {
  const int M = argc > 1 ? atoi(argv[1]) : 256 * 197;
  const int reps = argc > 2 ? atoi(argv[2]) : 15;
  struct Shape { const char* name; int N, K; } shapes[] = {{"qkv", 2304, 768}, {"o", 768, 768}, {"fc1", 3072, 768}, {"fc2", 768, 3072}};
  std::vector<Cand> cands = {
      make<G<128, 256, 32, 64, 64, 3, 8>>("t128x256x32_w64x64_s3"),   // the tile cuBLASLt picks
      make<G<128, 256, 32, 64, 64, 4, 8>>("t128x256x32_w64x64_s4"),
      make<G<256, 128, 32, 64, 64, 3, 8>>("t256x128x32_w64x64_s3"),
      make<G<256, 128, 32, 64, 64, 4, 8>>("t256x128x32_w64x64_s4"),
      make<G<128, 128, 32, 64, 64, 4, 8>>("t128x128x32_w64x64_s4"),
      make<G<128, 128, 32, 64, 64, 6, 8>>("t128x128x32_w64x64_s6"),
      make<G<128, 128, 64, 64, 64, 3, 8>>("t128x128x64_w64x64_s3"),
      make<G<128, 256, 32, 64, 64, 3, 4>>("t128x256x32_w64x64_s3_sw4"),
      make<G<128, 256, 32, 64, 64, 3, 16>>("t128x256x32_w64x64_s3_sw16"),
      make<G<64, 256, 32, 64, 64, 4, 8>>("t64x256x32_w64x64_s4"),
  };
  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  Lt lt;
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  printf("{\"kind\":\"env\",\"gpu\":\"%s\",\"sms\":%d,\"smem_optin\":%zu,\"M\":%d}\n", p.name, p.multiProcessorCount,
         p.sharedMemPerBlockOptin, M);
  for (auto& s : shapes) {
    const long nA = (long)M * s.K, nW = (long)s.N * s.K, nD = (long)M * s.N;
    __nv_bfloat16 *A, *W, *b, *Dref, *D;
    CK(cudaMalloc(&A, nA * 2)); CK(cudaMalloc(&W, nW * 2)); CK(cudaMalloc(&b, s.N * 2));
    CK(cudaMalloc(&Dref, nD * 2)); CK(cudaMalloc(&D, nD * 2));
    fill<<<1024, 256, 0, st>>>(A, nA, 1, 1.f);
    fill<<<1024, 256, 0, st>>>(W, nW, 2, 1.f / sqrtf((float)s.K));
    fill<<<64, 256, 0, st>>>(b, s.N, 3, 0.1f);
    float* dm;
    CK(cudaMalloc(&dm, 8));
    const double gf = 2.0 * M * s.N * s.K / 1e9;
    const int iters = std::max(5, (int)(200.0 / (gf / 300.0)));  // ~200 ms per rep... capped below
    const int it = std::min(iters, 50);
    auto report = [&](const char* name, std::vector<double> v, float maxdiff, float scale, int rc) {
      const double med = v[v.size() / 2];
      printf("{\"kind\":\"gemm\",\"shape\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"impl\":\"%s\",\"rc\":%d,\"ms\":%.5f,"
             "\"ms_lo\":%.5f,\"ms_hi\":%.5f,\"tflops\":%.1f,\"pct_peak\":%.1f,\"max_abs_diff\":%.4g,\"ref_absmax\":%.4g}\n",
             s.name, M, s.N, s.K, name, rc, med, v[v.size() / 4], v[3 * v.size() / 4], gf / med, 100 * gf / med / PEAK,
             maxdiff, scale);
      fflush(stdout);
    };
    auto vref = timeit([&] { lt.run(Dref, A, W, b, M, s.N, s.K, st); }, st, reps, it);
    report("cublaslt", vref, 0.f, 0.f, 0);
    for (auto& c : cands) {
      CK(cudaMemsetAsync(D, 0, nD * 2, st));
      int rc = c.fn(D, A, W, b, M, s.N, s.K, st);
      CK(cudaStreamSynchronize(st));
      if (cudaGetLastError() != cudaSuccess) rc = -9;
      if (rc) { report(c.name.c_str(), {0.0}, -1.f, 0.f, rc); continue; }
      CK(cudaMemsetAsync(dm, 0, 8, st));
      diff<<<1024, 256, 0, st>>>(D, Dref, nD, dm);
      float h[2];
      CK(cudaMemcpy(h, dm, 8, cudaMemcpyDeviceToHost));
      auto v = timeit([&] { c.fn(D, A, W, b, M, s.N, s.K, st); }, st, reps, it);
      report(c.name.c_str(), v, h[0], h[1], 0);
    }
    cudaFree(A); cudaFree(W); cudaFree(b); cudaFree(Dref); cudaFree(D); cudaFree(dm);
  }
  return 0;
}
