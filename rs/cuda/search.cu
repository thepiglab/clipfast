// Embedding search: int8 coarse scoring ("low") + exact fp32 re-rank of the top candidates ("high").
// Corpus: int8 with ONE global scale (99.9th percentile of per-vector max|x| / 127); queries: int8 with a per-query
// scale. Scores: dp4a kernel for <= 8 queries, cuBLAS int8 GEMM (int32 accumulate) otherwise, in 65536-column chunks.
// Per-query top-kc via block radix sorts (segment top-kc, then a running merge). The kc candidates are re-scored exactly
// in fp32 from a device copy ("speed") or a mapped pinned host copy ("low memory"), and the final top-k is sorted.
// An exact fp32 brute-force mode (cuBLAS SGEMM) gives the reference. Top-k selection and the dp4a kernel are adapted
// from flashcluster (MIT, https://github.com/thepiglab/flashcluster).
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" int cs_cutlass_int8_gemm(const void* A, const void* B, void* D, int m, int n, int k, float alpha, cudaStream_t st);
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return -1; } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS %s:%d %d\n", __FILE__, __LINE__, (int)s_); return -1; } } while (0)

namespace {
constexpr int ST = 256, SI = 16, SEG = ST * SI;   // 4096 keys per block sort
constexpr int CHUNK = 16 * SEG;                   // 65536 corpus columns per GEMM chunk
using Sort = cub::BlockRadixSort<float, ST, SI, int32_t>;

// per-row max |x|
__global__ void row_maxabs(const float* __restrict__ x, int64_t n, int d, float* __restrict__ out) {
  const int64_t r = blockIdx.x * (int64_t)(blockDim.x / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  if (r >= n) return;
  float m = 0.f;
  for (int t = lane; t < d; t += 32) m = fmaxf(m, fabsf(x[r * d + t]));
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if (!lane) out[r] = m;
}
__global__ void quantize_fixed(const float* __restrict__ x, int64_t nd, float inv, int8_t* __restrict__ out) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < nd; i += (int64_t)gridDim.x * blockDim.x)
    out[i] = (int8_t)max(-127, min(127, __float2int_rn(x[i] * inv)));
}
// queries: per-row scale s = max|q| / 127, q8 = round(q / s)
__global__ void quantize_rows(const float* __restrict__ q, int nq, int d, int8_t* __restrict__ q8, float* __restrict__ qs) {
  const int r = blockIdx.x, lane = threadIdx.x;
  float m = 0.f;
  for (int t = lane; t < d; t += 32) m = fmaxf(m, fabsf(q[(int64_t)r * d + t]));
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  const float s = m > 0.f ? m / 127.f : 1.f;
  for (int t = lane; t < d; t += 32) q8[(int64_t)r * d + t] = (int8_t)max(-127, min(127, __float2int_rn(q[(int64_t)r * d + t] / s)));
  if (!lane) qs[r] = s;
}
// small batches: one warp per corpus row, 16-byte loads, dp4a; writes scores[q][c] = int * qs[q] * sg
template <int NQ>
__global__ void int8_gemv(const int8_t* __restrict__ X, int64_t c0, int cols, int d, const int8_t* __restrict__ Q,
                          int nq, float sg, const float* __restrict__ qs, float* __restrict__ out) {
  extern __shared__ int8_t sq[];
  for (int i = threadIdx.x; i < nq * d; i += blockDim.x) sq[i] = Q[i];
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  const int c = blockIdx.x * (blockDim.x / 32) + warp;
  if (c >= cols) return;
  const int4* x4 = reinterpret_cast<const int4*>(X + (c0 + c) * d);
  int acc[NQ];
#pragma unroll
  for (int q = 0; q < NQ; q++) acc[q] = 0;
  for (int t = lane; t < d / 16; t += 32) {
    const int4 a = x4[t];
#pragma unroll
    for (int q = 0; q < NQ; q++) {
      if (q >= nq) break;
      const int4 b = reinterpret_cast<const int4*>(sq + q * d)[t];
      acc[q] = __dp4a(a.x, b.x, acc[q]);
      acc[q] = __dp4a(a.y, b.y, acc[q]);
      acc[q] = __dp4a(a.z, b.z, acc[q]);
      acc[q] = __dp4a(a.w, b.w, acc[q]);
    }
  }
#pragma unroll
  for (int q = 0; q < NQ; q++) {
    if (q >= nq) break;
    int v = acc[q];
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (!lane) out[(int64_t)q * cols + c] = (float)v * qs[q] * sg;
  }
}
// f16 scores [nq][ld] (= int8 dot * alpha) -> f32 scores [nq][cols] * qs[q] * mult
__global__ void apply_scales16(const __half* __restrict__ s, int ld, float* __restrict__ out, int nq, int cols, float mult,
                               const float* __restrict__ qs) {
  const int64_t n = (int64_t)nq * cols;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t q = i / cols, c = i % cols;
    out[i] = __half2float(s[q * ld + c]) * qs[q] * mult;
  }
}
// int32 scores [nq][ld] (ld >= cols) -> f32 scores [nq][cols] * qs[q] * sg
__global__ void apply_scales(const int32_t* __restrict__ s, int ld, float* __restrict__ out, int nq, int cols, float sg,
                             const float* __restrict__ qs) {
  const int64_t n = (int64_t)nq * cols;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t q = i / cols, c = i % cols;
    out[i] = (float)s[q * ld + c] * qs[q] * sg;
  }
}
// stage 1: block (row r, segment g): top-k of scores[r, g*SEG ..) -> cand[r][g][0..k)
__global__ void __launch_bounds__(ST) seg_topk(const float* __restrict__ scores, int cols, int64_t base, int k,
                                               float* __restrict__ cs, int32_t* __restrict__ ci) {
  __shared__ typename Sort::TempStorage tmp;
  const int r = blockIdx.x, g = blockIdx.y, nseg = gridDim.y;
  float key[SI];
  int32_t val[SI];
  const float* row = scores + (int64_t)r * cols;
#pragma unroll
  for (int i = 0; i < SI; ++i) {
    const int c = g * SEG + threadIdx.x * SI + i;
    const bool ok = c < cols;
    key[i] = ok ? row[c] : -INFINITY;
    val[i] = ok ? (int32_t)(base + c) : INT32_MAX;
  }
  Sort(tmp).SortDescendingBlockedToStriped(key, val);
#pragma unroll
  for (int i = 0; i < SI; ++i) {
    const int rank = i * ST + threadIdx.x;
    if (rank < k) {
      cs[((int64_t)r * nseg + g) * k + rank] = key[i];
      ci[((int64_t)r * nseg + g) * k + rank] = val[i];
    }
  }
}
// merge, block r: running top-k + ncand candidates -> new running top-k (k + ncand <= SEG)
__global__ void __launch_bounds__(ST) merge_topk(const float* __restrict__ cs, const int32_t* __restrict__ ci, int ncand,
                                                 int k, float* __restrict__ run_s, int32_t* __restrict__ run_i) {
  __shared__ typename Sort::TempStorage tmp;
  const int r = blockIdx.x;
  float key[SI];
  int32_t val[SI];
#pragma unroll
  for (int i = 0; i < SI; ++i) {
    const int slot = threadIdx.x * SI + i;
    if (slot < k) {
      key[i] = run_s[(int64_t)r * k + slot];
      val[i] = run_i[(int64_t)r * k + slot];
    } else if (slot - k < ncand) {
      key[i] = cs[(int64_t)r * ncand + slot - k];
      val[i] = ci[(int64_t)r * ncand + slot - k];
    } else {
      key[i] = -INFINITY;
      val[i] = INT32_MAX;
    }
  }
  Sort(tmp).SortDescendingBlockedToStriped(key, val);
#pragma unroll
  for (int i = 0; i < SI; ++i) {
    const int rank = i * ST + threadIdx.x;
    if (rank < k) {
      run_s[(int64_t)r * k + rank] = key[i];
      run_i[(int64_t)r * k + rank] = val[i];
    }
  }
}
__global__ void fill(float* __restrict__ s, int32_t* __restrict__ i, int64_t n) {
  for (int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; j < n; j += (int64_t)gridDim.x * blockDim.x) {
    s[j] = -INFINITY;
    i[j] = INT32_MAX;
  }
}
// exact fp32 re-score of each query's kc candidates; one warp per candidate; X may be device or mapped host memory
__global__ void rescore_f32(const float* __restrict__ X, const float* __restrict__ Q, int d, int64_t n,
                            const int32_t* __restrict__ ids, int kc, float* __restrict__ out) {
  const int r = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x & 31, nw = blockDim.x / 32;
  for (int i = warp; i < kc; i += nw) {
    const int32_t id = ids[(int64_t)r * kc + i];
    float s = -INFINITY;
    if (id >= 0 && id < n) {
      s = 0.f;
      for (int t = lane; t < d; t += 32) s = fmaf(Q[(int64_t)r * d + t], X[(int64_t)id * d + t], s);
      for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    }
    if (!lane) out[(int64_t)r * kc + i] = s;
  }
}
// final: sort the kc re-scored candidates of row r, keep top k
__global__ void __launch_bounds__(128) final_topk(const float* __restrict__ cs, const int32_t* __restrict__ ci, int kc,
                                                  int k, float* __restrict__ os, int32_t* __restrict__ oi) {
  using S = cub::BlockRadixSort<float, 128, 8, int32_t>;
  __shared__ typename S::TempStorage tmp;
  const int r = blockIdx.x;
  float key[8];
  int32_t val[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int slot = threadIdx.x * 8 + i;
    key[i] = slot < kc ? cs[(int64_t)r * kc + slot] : -INFINITY;
    val[i] = slot < kc ? ci[(int64_t)r * kc + slot] : INT32_MAX;
  }
  S(tmp).SortDescendingBlockedToStriped(key, val);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rank = i * 128 + threadIdx.x;
    if (rank < k) {
      os[(int64_t)r * k + rank] = key[i];
      oi[(int64_t)r * k + rank] = val[i];
    }
  }
}
inline int nb(int64_t n, int t = 256) { return (int)std::min<int64_t>((n + t - 1) / t, 65535L * 16); }
}  // namespace

struct CS {
  int64_t n = 0;
  int d = 0;
  int rerank = 1;            // 1 device f32, 2 mapped pinned host f32
  float sg = 1.f;            // corpus int8 scale
  int8_t* x8 = nullptr;      // [n, d] int8 (device)
  float* xf = nullptr;       // [n, d] f32: device (rerank 1) or host pinned mapped (rerank 2)
  float* xf_dev = nullptr;   // device-visible pointer to xf
  cublasHandle_t blas = nullptr;
  cudaStream_t st = nullptr;
  // per-search workspaces (grown on demand)
  size_t cap_q = 0, cap_kc = 0;
  float *q = nullptr, *qs = nullptr, *scores = nullptr, *cs = nullptr, *run_s = nullptr, *res = nullptr, *outs = nullptr;
  int8_t* q8 = nullptr;
  int32_t *iscores = nullptr, *ci = nullptr, *run_i = nullptr, *outi = nullptr;
  size_t dev_bytes = 0, host_bytes = 0, ws_bytes = 0;
};

static int ensure(CS* h, int nq, int kc, int k) {
  const size_t need_kc = (size_t)std::max(kc, k);
  if ((size_t)nq <= h->cap_q && need_kc <= h->cap_kc) return 0;
  nq = std::max<int>(nq, (int)h->cap_q);
  kc = std::max<int>((int)need_kc, (int)h->cap_kc);
  for (void* p : {(void*)h->q, (void*)h->qs, (void*)h->scores, (void*)h->cs, (void*)h->run_s, (void*)h->res,
                  (void*)h->outs, (void*)h->q8, (void*)h->iscores, (void*)h->ci, (void*)h->run_i, (void*)h->outi})
    if (p) cudaFree(p);
  const size_t q = (size_t)nq, d = h->d, nseg = CHUNK / SEG;
  size_t tot = 0;
  auto A = [&](void** p, size_t b) { tot += b; return cudaMalloc(p, b) == cudaSuccess; };
  const int KC = std::max(kc, k);
  bool ok = A((void**)&h->q, q * d * 4) && A((void**)&h->qs, q * 4) && A((void**)&h->q8, q * d) &&
            A((void**)&h->scores, q * CHUNK * 4) && A((void**)&h->iscores, q * CHUNK * 4) &&
            A((void**)&h->cs, q * nseg * KC * 4) && A((void**)&h->ci, q * nseg * KC * 4) &&
            A((void**)&h->run_s, q * KC * 4) && A((void**)&h->run_i, q * KC * 4) && A((void**)&h->res, q * KC * 4) &&
            A((void**)&h->outs, q * KC * 4) && A((void**)&h->outi, q * KC * 4);
  if (!ok) return -1;
  h->cap_q = q;
  h->cap_kc = (size_t)KC;
  h->ws_bytes = tot;
  return 0;
}

extern "C" {

// emb: host f32 [n, d] (rows L2-normalized embeddings, any d % 16 == 0). rerank: 1 = device f32 copy (speed),
// 2 = pinned host f32 copy read through mapped memory (low device memory).
CS* cs_create(const float* emb, int64_t n, int d, int rerank) {
  if (d % 16 || n <= 0 || n > INT32_MAX - 8) return nullptr;
  CS* h = new CS();
  h->n = n;
  h->d = d;
  h->rerank = rerank;
  if (cublasCreate(&h->blas) != CUBLAS_STATUS_SUCCESS || cudaStreamCreateWithFlags(&h->st, cudaStreamNonBlocking) != cudaSuccess)
    return nullptr;
  cublasSetStream(h->blas, h->st);
  const size_t nd = (size_t)n * d;
  // f32 copy for exact re-rank
  if (rerank == 2) {
    if (cudaHostAlloc((void**)&h->xf, nd * 4, cudaHostAllocMapped) != cudaSuccess) return nullptr;
    memcpy(h->xf, emb, nd * 4);
    if (cudaHostGetDevicePointer((void**)&h->xf_dev, h->xf, 0) != cudaSuccess) return nullptr;
    h->host_bytes = nd * 4;
  } else {
    if (cudaMalloc((void**)&h->xf, nd * 4) != cudaSuccess) return nullptr;
    cudaMemcpy(h->xf, emb, nd * 4, cudaMemcpyHostToDevice);
    h->xf_dev = h->xf;
    h->dev_bytes += nd * 4;
  }
  // int8 corpus with one global scale: 99.9th percentile of per-row max |x|, / 127
  if (cudaMalloc((void**)&h->x8, (nd + 16 * (size_t)d)) != cudaSuccess) return nullptr;
  h->dev_bytes += nd + 16 * (size_t)d;
  cudaMemset(h->x8 + nd, 0, 16 * (size_t)d);
  float* mx;
  float* tmp;
  const int64_t CH = 1 << 20;
  if (cudaMalloc((void**)&mx, n * 4) != cudaSuccess || cudaMalloc((void**)&tmp, std::min<int64_t>(CH, n) * d * 4) != cudaSuccess)
    return nullptr;
  for (int64_t s0 = 0; s0 < n; s0 += CH) {
    const int64_t m = std::min(CH, n - s0);
    const float* src = h->xf_dev ? h->xf_dev + s0 * d : nullptr;
    if (rerank == 2) {  // stage host rows through device memory
      cudaMemcpy(tmp, emb + s0 * d, m * d * 4, cudaMemcpyHostToDevice);
      src = tmp;
    }
    row_maxabs<<<(unsigned)((m + 7) / 8), 256>>>(src, m, d, mx + s0);
  }
  std::vector<float> hm(n);
  cudaMemcpy(hm.data(), mx, n * 4, cudaMemcpyDeviceToHost);
  const size_t p = std::min<size_t>(n - 1, (size_t)std::ceil(0.999 * (n - 1)));
  std::nth_element(hm.begin(), hm.begin() + p, hm.end());
  h->sg = std::max(hm[p], 1e-12f) / 127.f;
  for (int64_t s0 = 0; s0 < n; s0 += CH) {
    const int64_t m = std::min(CH, n - s0);
    const float* src = h->xf_dev + s0 * d;
    if (rerank == 2) {
      cudaMemcpy(tmp, emb + s0 * d, m * d * 4, cudaMemcpyHostToDevice);
      src = tmp;
    }
    quantize_fixed<<<nb(m * d), 256>>>(src, m * d, 1.f / h->sg, h->x8 + s0 * d);
  }
  cudaDeviceSynchronize();
  cudaFree(mx);
  cudaFree(tmp);
  return cudaGetLastError() == cudaSuccess ? h : nullptr;
}

// q: host f32 [nq, d]. Approximate int8 search + exact fp32 re-rank of the top kc (kc <= 4096 - 16*kc constraint:
// 16*kc + kc <= 4096 -> kc <= 240), or exact fp32 brute force when exact != 0. Writes top-k ids/scores (host).
int cs_search(CS* h, const float* qh, int nq, int k, int kc, int exact, int32_t* ids, float* scores) {
  const int nseg = CHUNK / SEG;
  if (exact) kc = k;
  if (kc < k || kc * (nseg + 1) > SEG || k > 1024) return -2;
  if (ensure(h, nq, kc, k)) return -1;
  const int d = h->d;
  CK(cudaMemcpyAsync(h->q, qh, (size_t)nq * d * 4, cudaMemcpyHostToDevice, h->st));
  if (!exact) quantize_rows<<<nq, 32, 0, h->st>>>(h->q, nq, d, h->q8, h->qs);
  fill<<<nb((int64_t)nq * kc), 256, 0, h->st>>>(h->run_s, h->run_i, (int64_t)nq * kc);
  for (int64_t c0 = 0; c0 < h->n; c0 += CHUNK) {
    const int cols = (int)std::min<int64_t>(CHUNK, h->n - c0);
    if (exact) {
      const float one = 1.f, zero = 0.f;  // scores[q][c] = X[c] . Q[q]   (fp32, no TF32)
      CB(cublasSgemm(h->blas, CUBLAS_OP_T, CUBLAS_OP_N, cols, nq, d, &one, h->xf_dev + c0 * d, d, h->q, d, &zero,
                     h->scores, cols));
    } else if (nq <= 8 && d % 16 == 0) {
      int8_gemv<8><<<(cols + 7) / 8, 256, (size_t)nq * d, h->st>>>(h->x8, c0, cols, d, h->q8, nq, h->sg, h->qs, h->scores);
    } else {
      // CUTLASS int8 tensor-core GEMM (sm80 mma.sync, runs on sm_120): f16 = acc / 256 keeps int8 dots in range
      const int colsp = (cols + 7) & ~7;  // pad rows are zero
      const float alpha = 1.f / 256;
      const int rc = cs_cutlass_int8_gemm(h->x8 + c0 * d, h->q8, h->iscores, colsp, nq, d, alpha, h->st);
      if (rc == 0) {
        apply_scales16<<<nb((int64_t)nq * cols), 256, 0, h->st>>>((const __half*)h->iscores, colsp, h->scores, nq, cols,
                                                                  h->sg / alpha, h->qs);
      } else {  // no CUTLASS: cuBLAS int8 GEMM (int32 accumulate)
        const int32_t one = 1, zero = 0;
        CB(cublasGemmEx(h->blas, CUBLAS_OP_T, CUBLAS_OP_N, colsp, nq, d, &one, h->x8 + c0 * d, CUDA_R_8I, d, h->q8,
                        CUDA_R_8I, d, &zero, h->iscores, CUDA_R_32I, colsp, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
        apply_scales<<<nb((int64_t)nq * cols), 256, 0, h->st>>>(h->iscores, colsp, h->scores, nq, cols, h->sg, h->qs);
      }
    }
    const int segs = (cols + SEG - 1) / SEG;
    seg_topk<<<dim3(nq, segs), ST, 0, h->st>>>(h->scores, cols, c0, kc, h->cs, h->ci);
    merge_topk<<<nq, ST, 0, h->st>>>(h->cs, h->ci, segs * kc, kc, h->run_s, h->run_i);
  }
  if (exact) {
    CK(cudaMemcpyAsync(scores, h->run_s, (size_t)nq * k * 4, cudaMemcpyDeviceToHost, h->st));
    CK(cudaMemcpyAsync(ids, h->run_i, (size_t)nq * k * 4, cudaMemcpyDeviceToHost, h->st));
  } else {
    rescore_f32<<<nq, 256, 0, h->st>>>(h->xf_dev, h->q, d, h->n, h->run_i, kc, h->res);
    final_topk<<<nq, 128, 0, h->st>>>(h->res, h->run_i, kc, k, h->outs, h->outi);
    CK(cudaMemcpyAsync(scores, h->outs, (size_t)nq * k * 4, cudaMemcpyDeviceToHost, h->st));
    CK(cudaMemcpyAsync(ids, h->outi, (size_t)nq * k * 4, cudaMemcpyDeviceToHost, h->st));
  }
  CK(cudaStreamSynchronize(h->st));
  return 0;
}

void cs_mem(CS* h, size_t* dev, size_t* host, size_t* ws, float* sg) {
  *dev = h->dev_bytes;
  *host = h->host_bytes;
  *ws = h->ws_bytes;
  *sg = h->sg;
}

void cs_destroy(CS* h) {
  if (!h) return;
  for (void* p : {(void*)h->q, (void*)h->qs, (void*)h->scores, (void*)h->cs, (void*)h->run_s, (void*)h->res,
                  (void*)h->outs, (void*)h->q8, (void*)h->iscores, (void*)h->ci, (void*)h->run_i, (void*)h->outi,
                  (void*)h->x8})
    if (p) cudaFree(p);
  if (h->rerank == 2) cudaFreeHost(h->xf); else cudaFree(h->xf);
  cublasDestroy(h->blas);
  cudaStreamDestroy(h->st);
  delete h;
}
}
