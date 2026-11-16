// clipfast-rs GPU engine: BiomedCLIP (ViT-B/16 + PubMedBERT-base) forward, bf16 (fast) or fp32 (parity) mode.
// All buffers are preallocated in cf_create; forward passes allocate nothing and are captured into one CUDA graph
// per (tower, batch bucket, length bucket). C API at the bottom (called from Rust).
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#define CK(x)                                                                                      \
  do {                                                                                             \
    cudaError_t e_ = (x);                                                                          \
    if (e_ != cudaSuccess) {                                                                       \
      fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_));   \
      return -1;                                                                                   \
    }                                                                                              \
  } while (0)
#define CB(x)                                                                                      \
  do {                                                                                             \
    cublasStatus_t s_ = (x);                                                                       \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                             \
      fprintf(stderr, "cuBLASLt %s at %s:%d: %d\n", #x, __FILE__, __LINE__, (int)s_);               \
      return -1;                                                                                   \
    }                                                                                              \
  } while (0)

namespace {

constexpr int C = 768, H = 12, D = 64, FF = 3072, E = 512, P1 = 640, NTOK = 197, NPATCH = 196, LAYERS = 12;

template <typename T> __device__ __forceinline__ float tof(T v);
template <> __device__ __forceinline__ float tof<float>(float v) { return v; }
template <> __device__ __forceinline__ float tof<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T fromf(float v);
template <> __device__ __forceinline__ float fromf<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 fromf<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// ---------------- elementwise / small kernels ----------------

// pixels [B,3,224,224] f32 -> patches [B*196, 768] in (c, ky, kx) order
template <typename T>
__global__ void k_patchify(const float* __restrict__ px, T* __restrict__ out, int B) {
  const long total = (long)B * NPATCH * C;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int col = i % C;
    const long row = i / C;
    const int b = row / NPATCH, p = row % NPATCH, py = p / 14, pxx = p % 14;
    const int c = col / 256, ky = (col / 16) % 16, kx = col % 16;
    out[i] = fromf<T>(px[(((long)b * 3 + c) * 224 + py * 16 + ky) * 224 + pxx * 16 + kx]);
  }
}

// x[b,0] = cls_pos0 ; x[b,1+t] = tmp[b,t] + pos[t]
template <typename T>
__global__ void k_assemble(const T* __restrict__ tmp, const T* __restrict__ clspos0, const T* __restrict__ pos,
                           T* __restrict__ x, int B) {
  const long total = (long)B * NTOK * C;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int col = i % C;
    const long row = i / C;
    const int b = row / NTOK, t = row % NTOK;
    float v = t == 0 ? tof(clspos0[col]) : tof(tmp[((long)b * NPATCH + t - 1) * C + col]) + tof(pos[(t - 1) * C + col]);
    x[i] = fromf<T>(v);
  }
}

// x[b,l] = word[ids[b,l]] + pos_type[l]
template <typename T>
__global__ void k_embed(const int* __restrict__ ids, const T* __restrict__ word, const T* __restrict__ postype,
                        T* __restrict__ x, int B, int L) {
  const long total = (long)B * L * C;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int col = i % C;
    const long row = i / C;
    const int l = row % L;
    x[i] = fromf<T>(tof(word[(long)ids[row] * C + col]) + tof(postype[(long)l * C + col]));
  }
}

// packed: x[i] = word[ids[i]] + pos_type[pos[i]]
template <typename T>
__global__ void k_embed_packed(const int* __restrict__ ids, const int* __restrict__ pos, const T* __restrict__ word,
                               const T* __restrict__ postype, T* __restrict__ x, long T_) {
  const long total = T_ * C;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int col = i % C;
    const long row = i / C;
    x[i] = fromf<T>(tof(word[(long)ids[row] * C + col]) + tof(postype[(long)pos[row] * C + col]));
  }
}

// out[b] = in[cu[b]] (CLS rows of packed sequences -> compact)
template <typename T>
__global__ void k_gather_cls(const T* __restrict__ in, const int* __restrict__ cu, T* __restrict__ out, int B) {
  const long total = (long)B * C;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int col = i % C, b = i / C;
    out[i] = in[(long)cu[b] * C + col];
  }
}

// LayerNorm over C=768, one warp per row; rows may be strided (CLS-only paths). In-place allowed.
template <typename T>
__global__ void k_layernorm(const T* __restrict__ in, long sin, T* __restrict__ out, long sout,
                            const T* __restrict__ w, const T* __restrict__ b, float eps, int rows) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= rows) return;
  const T* r = in + warp * sin;
  float v[24];
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 24; j++) { v[j] = tof(r[lane + 32 * j]); s += v[j]; }
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const float mean = s * (1.f / C);
  float q = 0.f;
#pragma unroll
  for (int j = 0; j < 24; j++) { const float d = v[j] - mean; q += d * d; }
#pragma unroll
  for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
  const float rstd = rsqrtf(q * (1.f / C) + eps);
  T* ro = out + warp * sout;
#pragma unroll
  for (int j = 0; j < 24; j++) {
    const int c = lane + 32 * j;
    ro[c] = fromf<T>((v[j] - mean) * rstd * tof(w[c]) + tof(b[c]));
  }
}

__global__ void k_gelu_bf16x8(__nv_bfloat16* __restrict__ x, long n8) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n8; i += (long)gridDim.x * blockDim.x) {
    uint4 v = reinterpret_cast<uint4*>(x)[i];
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 f = __bfloat1622float2(h[j]);
      f.x = 0.5f * f.x * (1.f + erff(f.x * 0.70710678118654752f));
      f.y = 0.5f * f.y * (1.f + erff(f.y * 0.70710678118654752f));
      h[j] = __floats2bfloat162_rn(f.x, f.y);
    }
    reinterpret_cast<uint4*>(x)[i] = v;
  }
}

template <typename T>
__global__ void k_gelu(T* __restrict__ x, long n) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    const float v = tof(x[i]);
    x[i] = fromf<T>(0.5f * v * (1.f + erff(v * 0.70710678118654752f)));
  }
}

// out[b] = normalize(in[b]) as float
template <typename T>
__global__ void k_normalize(const T* __restrict__ in, float* __restrict__ out, int B) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= B) return;
  float v[16], s = 0.f;
#pragma unroll
  for (int j = 0; j < 16; j++) { v[j] = tof(in[(long)warp * E + lane + 32 * j]); s += v[j] * v[j]; }
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const float inv = 1.f / fmaxf(sqrtf(s), 1e-12f);
#pragma unroll
  for (int j = 0; j < 16; j++) out[(long)warp * E + lane + 32 * j] = v[j] * inv;
}

// ---------------- attention ----------------
// qkv rows: [B, N, 3, H, D] (row stride 3C), out rows: [B, N, H*D] (row stride C).
// lens[b]: valid keys of sequence b (keys >= len masked). nq: number of query rows to compute (N, or 1 for CLS).

// fp32 / generic reference-quality kernel: one warp per (b, h, query row); scores in shared memory.
template <typename T>
__global__ void k_attn_simple(const T* __restrict__ qkv, T* __restrict__ out, const int* __restrict__ lens,
                              const int* __restrict__ cu, int N, int nq, float scale) {
  extern __shared__ float sm[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wpb = blockDim.x >> 5;
  const int qi = blockIdx.x * wpb + warp, h = blockIdx.y, b = blockIdx.z;
  float* s = sm + warp * N;
  const long r0 = cu ? cu[b] : (long)b * N;  // first row of sequence b
  const int len = cu ? cu[b + 1] - cu[b] : (lens ? lens[b] : N);
  const int qlim = cu ? min(nq, len) : nq;
  if (qi >= qlim) return;
  const T* q = qkv + (r0 + qi) * 3 * C + h * D;
  const float q0 = tof(q[lane]), q1 = tof(q[lane + 32]);
  float mx = -INFINITY;
  for (int j = 0; j < len; j++) {
    const T* k = qkv + (r0 + j) * 3 * C + C + h * D;
    float d = q0 * tof(k[lane]) + q1 * tof(k[lane + 32]);
#pragma unroll
    for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
    d *= scale;
    if (lane == 0) s[j] = d;
    mx = fmaxf(mx, d);
  }
  __syncwarp();
  float den = 0.f;
  for (int j = lane; j < len; j += 32) { const float e = expf(s[j] - mx); s[j] = e; den += e; }
#pragma unroll
  for (int o = 16; o; o >>= 1) den += __shfl_xor_sync(0xffffffffu, den, o);
  __syncwarp();
  float a0 = 0.f, a1 = 0.f;
  for (int j = 0; j < len; j++) {
    const T* v = qkv + (r0 + j) * 3 * C + 2 * C + h * D;
    a0 += s[j] * tof(v[lane]);
    a1 += s[j] * tof(v[lane + 32]);
  }
  T* o = out + (r0 + qi) * C + h * D;
  o[lane] = fromf<T>(a0 / den);
  o[lane + 32] = fromf<T>(a1 / den);
}

// bf16 flash-style kernel (FA2 structure): CTA = 4 warps = 64 query rows of one (b, h); stream 64-key blocks.
__device__ __forceinline__ void mma_bf16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4(unsigned* r, const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned* r, const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<unsigned*>(&v);
}

constexpr int AB = 64, APAD = 72;  // block rows, padded smem row (elements) to avoid bank conflicts

__global__ void __launch_bounds__(128) k_attn_bf16(const __nv_bfloat16* __restrict__ qkv,
                                                   __nv_bfloat16* __restrict__ out, const int* __restrict__ lens,
                                                   const int* __restrict__ cu, int N, int nq, float scale_log2) {
  __shared__ __align__(16) __nv_bfloat16 sQ[AB * APAD], sK[AB * APAD], sV[AB * APAD];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int q0 = blockIdx.x * AB, h = blockIdx.y, b = blockIdx.z;
  const long rowb = cu ? cu[b] : (long)b * N;  // first row of sequence b
  const int len = cu ? cu[b + 1] - cu[b] : (lens ? lens[b] : N);
  if (cu) nq = min(nq, len);
  if (q0 >= nq) return;  // whole CTA exits together
  const long base = rowb * 3 * C;
  // load Q tile (rows >= nq -> 0)
  for (int i = tid; i < AB * 8; i += 128) {
    const int r = i >> 3, c8 = (i & 7) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (q0 + r < nq) v = *reinterpret_cast<const uint4*>(qkv + base + (long)(q0 + r) * 3 * C + h * D + c8);
    *reinterpret_cast<uint4*>(sQ + r * APAD + c8) = v;
  }
  __syncthreads();
  unsigned qa[4][4];  // A fragments for k = 0..63 (4 k16 steps), rows warp*16..+16
#pragma unroll
  for (int ks = 0; ks < 4; ks++)
    ldsm_x4(qa[ks], sQ + (warp * 16 + (lane & 15)) * APAD + ks * 16 + (lane >> 4) * 8);
  float o[8][4];
#pragma unroll
  for (int i = 0; i < 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  for (int k0 = 0; k0 < len; k0 += AB) {
    __syncthreads();
    for (int i = tid; i < AB * 8; i += 128) {
      const int r = i >> 3, c8 = (i & 7) * 8;
      uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
      if (k0 + r < len) {
        const __nv_bfloat16* rowp = qkv + base + (long)(k0 + r) * 3 * C + h * D + c8;
        kv = *reinterpret_cast<const uint4*>(rowp + C);
        vv = *reinterpret_cast<const uint4*>(rowp + 2 * C);
      }
      *reinterpret_cast<uint4*>(sK + r * APAD + c8) = kv;
      *reinterpret_cast<uint4*>(sV + r * APAD + c8) = vv;
    }
    __syncthreads();
    float s[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
#pragma unroll
      for (int nt = 0; nt < 8; nt += 2) {  // two key n8-tiles per ldmatrix.x4
        unsigned kb[4];
        ldsm_x4(kb, sK + (nt * 8 + (lane & 7) + ((lane >> 4) << 3)) * APAD + ks * 16 + ((lane >> 3) & 1) * 8);
        mma_bf16(s[nt], qa[ks], kb[0], kb[1]);
        mma_bf16(s[nt + 1], qa[ks], kb[2], kb[3]);
      }
    }
    // mask, online softmax (rows g and g+8 of this warp's 16)
    float mx0 = m0, mx1 = m1;
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
#pragma unroll
      for (int j = 0; j < 2; j++) {
        const int key = k0 + nt * 8 + 2 * t + j;
        s[nt][j] = key < len ? s[nt][j] * scale_log2 : -INFINITY;
        s[nt][2 + j] = key < len ? s[nt][2 + j] * scale_log2 : -INFINITY;
        mx0 = fmaxf(mx0, s[nt][j]);
        mx1 = fmaxf(mx1, s[nt][2 + j]);
      }
    }
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
    }
    const float c0 = exp2f(m0 - mx0), c1 = exp2f(m1 - mx1);
    m0 = mx0;
    m1 = mx1;
    float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      s[nt][0] = exp2f(s[nt][0] - mx0);
      s[nt][1] = exp2f(s[nt][1] - mx0);
      s[nt][2] = exp2f(s[nt][2] - mx1);
      s[nt][3] = exp2f(s[nt][3] - mx1);
      rs0 += s[nt][0] + s[nt][1];
      rs1 += s[nt][2] + s[nt][3];
    }
    l0 = l0 * c0 + rs0;  // partial (per-lane) row sums; reduced across the quad at the end
    l1 = l1 * c1 + rs1;
#pragma unroll
    for (int dt = 0; dt < 8; dt++) {
      o[dt][0] *= c0;
      o[dt][1] *= c0;
      o[dt][2] *= c1;
      o[dt][3] *= c1;
    }
    // O += P V : P (16 x 64 keys) from S accumulators, V via ldmatrix.trans
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {  // 16 keys per step
      unsigned pa[4];
      pa[0] = pack_bf16(s[2 * kk][0], s[2 * kk][1]);
      pa[1] = pack_bf16(s[2 * kk][2], s[2 * kk][3]);
      pa[2] = pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]);
      pa[3] = pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
      for (int dt = 0; dt < 8; dt += 2) {
        unsigned vb[4];
        ldsm_x4_t(vb, sV + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * APAD + dt * 8 + (lane >> 4) * 8);
        mma_bf16(o[dt], pa, vb[0], vb[1]);
        mma_bf16(o[dt + 1], pa, vb[2], vb[3]);
      }
    }
  }
#pragma unroll
  for (int off = 1; off < 4; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  const float i0 = 1.f / l0, i1 = 1.f / l1;
  const int r0 = q0 + warp * 16 + g, r1 = r0 + 8;
#pragma unroll
  for (int dt = 0; dt < 8; dt++) {
    const int col = h * D + dt * 8 + 2 * t;
    if (r0 < nq)
      *reinterpret_cast<__nv_bfloat162*>(out + (rowb + r0) * C + col) =
          __floats2bfloat162_rn(o[dt][0] * i0, o[dt][1] * i0);
    if (r1 < nq)
      *reinterpret_cast<__nv_bfloat162*>(out + (rowb + r1) * C + col) =
          __floats2bfloat162_rn(o[dt][2] * i1, o[dt][3] * i1);
  }
}

inline int nblk(long n, int t = 256) { long b = (n + t - 1) / t; return (int)(b > 65535L * 8 ? 65535L * 8 : b); }

}  // namespace

// ---------------- engine ----------------
struct Lin { void* w = nullptr; void* b = nullptr; };
struct Blk { void *ln1w, *ln1b, *ln2w, *ln2b; Lin qkv, o, fc1, fc2; };

struct CF {
  bool bf16;
  size_t es;                       // element size
  cublasLtHandle_t lt;
  void* ws;
  size_t ws_bytes = 64 << 20;
  cudaStream_t st;
  std::map<std::string, void*> W;  // device weights by name
  Blk vb[LAYERS], tb[LAYERS];
  // activations
  int max_img, max_rows;           // image chunk (images), text rows (B*L)
  void *x, *h, *qkv, *a, *f, *tmp;
  float* px;                       // pixels input [max_img,3,224,224]
  int *ids, *lens, *pos, *cu;      // text input (padded: ids+lens; packed: ids+pos+cu)
  float* out;                      // [max(B),512] normalized embeddings
  void* emb;                       // [max(B),512] pre-normalize
  std::map<std::tuple<int, int, int>, cudaGraphExec_t> graphs;
  std::map<std::tuple<int, int, int, long, long, long, int>, cublasLtMatmulAlgo_t> algos;
  size_t weight_bytes = 0, act_bytes = 0;
  int gelu_epilogue = 0;
  int cutlass_gelu = 1;            // CUTLASS fc1 GEMM with fused exact-erf GELU (bf16, M >= 8192)
  int autotune = 0;                // opt-in: time top cuBLASLt heuristics per shape (r5 A/B: no gain, +0.4 GB scratch)
  void* scratch = nullptr;         // autotune output scratch (never live data)
  size_t scratch_bytes = 0;           // 1: cuBLASLt GELU_BIAS (tanh approx) instead of bias + exact-erf kernel
};

static void* WP(CF* cf, const std::string& n) {
  auto it = cf->W.find(n);
  if (it == cf->W.end()) { fprintf(stderr, "missing weight %s\n", n.c_str()); return nullptr; }
  return it->second;
}

// D[M,N] (row stride ldd) = A[M,K] (row stride lda) @ W[N,K]^T + bias (+ D if accumulate)
static int gemm(CF* cf, void* Dp, long ldd, const void* A, long lda, const void* Wt, const void* bias, int M, int N,
                int K, bool accumulate, bool gelu = false) {
  cudaDataType_t ty = cf->bf16 ? CUDA_R_16BF : CUDA_R_32F;
  cublasLtMatmulDesc_t op;
  CB(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  cublasLtEpilogue_t epi = gelu ? CUBLASLT_EPILOGUE_GELU_BIAS : (bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT);
  CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
  if (bias) CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  // column-major view: D^T [N x M] = W [N x K] (stored K x N col-major, op T) * A^T [K x M]
  cublasLtMatrixLayout_t la, lb, lc;
  CB(cublasLtMatrixLayoutCreate(&la, ty, K, N, K));
  CB(cublasLtMatrixLayoutCreate(&lb, ty, K, M, lda));
  CB(cublasLtMatrixLayoutCreate(&lc, ty, N, M, ldd));
  auto key = std::make_tuple(M, N, K, lda, ldd, (long)(bias != nullptr) + 2 * (long)gelu, (int)accumulate);
  auto it = cf->algos.find(key);
  cublasLtMatmulAlgo_t algo;
  if (it == cf->algos.end()) {
    cublasLtMatmulPreference_t pref;
    CB(cublasLtMatmulPreferenceCreate(&pref));
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cf->ws_bytes,
                                            sizeof(cf->ws_bytes)));
    cublasLtMatmulHeuristicResult_t res[16];
    int n = 0;
    CB(cublasLtMatmulAlgoGetHeuristic(cf->lt, op, la, lb, lc, lc, pref, cf->autotune ? 16 : 1, res, &n));
    cublasLtMatmulPreferenceDestroy(pref);
    if (n < 1) { fprintf(stderr, "no cuBLASLt algo for M%d N%d K%d\n", M, N, K); return -1; }
    algo = res[0].algo;
    cudaStreamCaptureStatus cs;
    cudaStreamIsCapturing(cf->st, &cs);
    const size_t need = (size_t)ldd * (M - 1) * cf->es + (size_t)N * cf->es;
    if (cf->autotune && n > 1 && cs == cudaStreamCaptureStatusNone && need <= cf->scratch_bytes) {
      // time each candidate on scratch output (C = D = scratch, so live activations are never touched)
      cudaEvent_t e0, e1;
      cudaEventCreate(&e0);
      cudaEventCreate(&e1);
      const float al = 1.f, be = accumulate ? 1.f : 0.f;
      float best = 1e30f;
      for (int c = 0; c < n; c++) {
        if (res[c].state != CUBLAS_STATUS_SUCCESS) continue;
        bool ok = true;
        for (int r = 0; r < 2 && ok; r++)  // warm
          ok = cublasLtMatmul(cf->lt, op, &al, Wt, la, A, lb, &be, cf->scratch, lc, cf->scratch, lc, &res[c].algo, cf->ws,
                              cf->ws_bytes, cf->st) == CUBLAS_STATUS_SUCCESS;
        if (!ok) continue;
        cudaEventRecord(e0, cf->st);
        for (int r = 0; r < 5; r++)
          cublasLtMatmul(cf->lt, op, &al, Wt, la, A, lb, &be, cf->scratch, lc, cf->scratch, lc, &res[c].algo, cf->ws,
                         cf->ws_bytes, cf->st);
        cudaEventRecord(e1, cf->st);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) { best = ms; algo = res[c].algo; }
      }
      cudaEventDestroy(e0);
      cudaEventDestroy(e1);
    }
    cf->algos[key] = algo;
  } else {
    algo = it->second;
  }
  const float alpha = 1.f, beta = accumulate ? 1.f : 0.f;
  CB(cublasLtMatmul(cf->lt, op, &alpha, Wt, la, A, lb, &beta, Dp, lc, Dp, lc, &algo, cf->ws, cf->ws_bytes, cf->st));
  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulDescDestroy(op);
  return 0;
}

#define BF(p) reinterpret_cast<__nv_bfloat16*>(const_cast<void*>((const void*)(p)))
#define FP(p) reinterpret_cast<float*>(const_cast<void*>((const void*)(p)))
#define LAUNCH(cf, KERNEL, grid, block, smem, ...)                                                    \
  do {                                                                                                 \
    if ((cf)->bf16) {                                                                                  \
      using T = __nv_bfloat16;                                                                         \
      KERNEL<T><<<grid, block, smem, (cf)->st>>>(__VA_ARGS__);                                         \
    } else {                                                                                           \
      using T = float;                                                                                 \
      KERNEL<T><<<grid, block, smem, (cf)->st>>>(__VA_ARGS__);                                         \
    }                                                                                                  \
  } while (0)
#define TP(p) reinterpret_cast<T*>(const_cast<void*>((const void*)(p)))
#define TRY(x) do { if ((x) != 0) return -1; } while (0)

static int gelu(CF* cf, void* x, long n);

extern "C" int cf_gemm_bias_gelu_bf16(void* D, const void* A, long lda, const void* W, const void* bias, int M, int N,
                                      int K, cudaStream_t st, int variant);

// fc1 + exact GELU. bf16 large M: CUTLASS GEMM with fused erf-GELU epilogue (one rounding, no extra pass);
// otherwise cuBLASLt (bias epilogue, or tanh GELU_BIAS if opted in) + the GELU kernel.
static int fc1_gelu(CF* cf, void* F_, const void* A, long lda, const void* W, const void* b, int M) {
  if (cf->bf16 && cf->cutlass_gelu && !cf->gelu_epilogue && M >= 8192) {
    const int rc = cf_gemm_bias_gelu_bf16(F_, A, lda, W, b, M, FF, C, cf->st, 0);
    if (rc == 0) return 0;
    fprintf(stderr, "cutlass gelu gemm unavailable (rc %d); falling back to cuBLASLt + GELU kernel\n", rc);
    cf->cutlass_gelu = 0;
  }
  TRY(gemm(cf, F_, FF, A, lda, W, b, M, FF, C, false, cf->gelu_epilogue));
  if (!cf->gelu_epilogue) TRY(gelu(cf, F_, (long)M * FF));
  return 0;
}

static int layernorm(CF* cf, const void* in, long sin, void* out, long sout, const void* w, const void* b, float eps,
                     int rows) {
  LAUNCH(cf, k_layernorm, (rows * 32 + 255) / 256, 256, 0, TP(in), sin, TP(out), sout, TP(w), TP(b), eps, rows);
  return 0;
}
static int gelu(CF* cf, void* x, long n) {
  if (cf->bf16 && n % 8 == 0) {
    k_gelu_bf16x8<<<nblk(n / 8), 256, 0, cf->st>>>(BF(x), n / 8);
    return 0;
  }
  LAUNCH(cf, k_gelu, nblk(n), 256, 0, TP(x), n);
  return 0;
}
// cu != nullptr: packed varlen sequences; N is then the max sequence length (grid size / smem), nq 1 = CLS only.
static int attention(CF* cf, const void* qkv, void* out, const int* lens, int B, int N, int nq,
                     const int* cu = nullptr) {
  const float scale = 0.125f;  // 1/sqrt(64)
  if (cf->bf16) {
    dim3 grid((std::min(nq, N) + AB - 1) / AB, H, B);
    k_attn_bf16<<<grid, 128, 0, cf->st>>>(BF(qkv), BF(out), lens, cu, N, nq, scale * 1.4426950408889634f);
  } else {
    const int wpb = 4;
    dim3 grid((std::min(nq, N) + wpb - 1) / wpb, H, B);
    k_attn_simple<float><<<grid, wpb * 32, wpb * N * sizeof(float), cf->st>>>(FP(qkv), FP(out), lens, cu, N, nq,
                                                                             scale);
  }
  return 0;
}

// ViT-B/16 (pre-norm). Input cf->px [B,3,224,224]. Output cf->out [B,512] normalized.
static int vision_forward(CF* cf, int B) {
  const long R = (long)B * NTOK, S = (long)NTOK * C;  // S: row stride between CLS rows
  LAUNCH(cf, k_patchify, nblk((long)B * NPATCH * C), 256, 0, cf->px, TP(cf->h), B);
  TRY(gemm(cf, cf->tmp, C, cf->h, C, WP(cf, "v.patch.w"), WP(cf, "v.patch.b"), B * NPATCH, C, C, false));
  LAUNCH(cf, k_assemble, nblk(R * C), 256, 0, TP(cf->tmp), TP(WP(cf, "v.cls_pos0")), TP(WP(cf, "v.pos")), TP(cf->x), B);
  for (int i = 0; i < LAYERS; i++) {
    const Blk& p = cf->vb[i];
    const bool last = i == LAYERS - 1;
    TRY(layernorm(cf, cf->x, C, cf->h, C, p.ln1w, p.ln1b, 1e-6f, (int)R));
    TRY(gemm(cf, cf->qkv, 3 * C, cf->h, C, p.qkv.w, p.qkv.b, (int)R, 3 * C, C, false));
    TRY(attention(cf, cf->qkv, cf->a, nullptr, B, NTOK, last ? 1 : NTOK));
    const int M = last ? B : (int)R;
    const long sx = last ? S : C;
    TRY(gemm(cf, cf->x, sx, cf->a, sx, p.o.w, p.o.b, M, C, C, true));
    TRY(layernorm(cf, cf->x, sx, cf->h, C, p.ln2w, p.ln2b, 1e-6f, M));
    TRY(fc1_gelu(cf, cf->f, cf->h, C, p.fc1.w, p.fc1.b, M));
    TRY(gemm(cf, cf->x, sx, cf->f, FF, p.fc2.w, p.fc2.b, M, C, FF, true));
  }
  TRY(layernorm(cf, cf->x, S, cf->h, C, WP(cf, "v.ln.w"), WP(cf, "v.ln.b"), 1e-6f, B));
  TRY(gemm(cf, cf->emb, E, cf->h, C, WP(cf, "v.proj.w"), nullptr, B, E, C, false));
  LAUNCH(cf, k_normalize, (B * 32 + 255) / 256, 256, 0, TP(cf->emb), cf->out, B);
  return 0;
}

// PubMedBERT (post-norm). Input cf->ids [B*L] int32 (padded), cf->lens [B]. Output cf->out [B,512] normalized.
static int text_forward(CF* cf, int B, int L) {
  const long R = (long)B * L, S = (long)L * C;
  const float eps = 1e-12f;
  LAUNCH(cf, k_embed, nblk(R * C), 256, 0, cf->ids, TP(WP(cf, "t.word")), TP(WP(cf, "t.pos_type")), TP(cf->x), B, L);
  TRY(layernorm(cf, cf->x, C, cf->x, C, WP(cf, "t.ln.w"), WP(cf, "t.ln.b"), eps, (int)R));
  for (int i = 0; i < LAYERS; i++) {
    const Blk& p = cf->tb[i];
    const bool last = i == LAYERS - 1;
    TRY(gemm(cf, cf->qkv, 3 * C, cf->x, C, p.qkv.w, p.qkv.b, (int)R, 3 * C, C, false));
    TRY(attention(cf, cf->qkv, cf->a, cf->lens, B, L, last ? 1 : L));
    const int M = last ? B : (int)R;
    const long sx = last ? S : C;
    TRY(gemm(cf, cf->x, sx, cf->a, sx, p.o.w, p.o.b, M, C, C, true));
    TRY(layernorm(cf, cf->x, sx, cf->x, sx, p.ln1w, p.ln1b, eps, M));
    TRY(fc1_gelu(cf, cf->f, cf->x, sx, p.fc1.w, p.fc1.b, M));
    TRY(gemm(cf, cf->x, sx, cf->f, FF, p.fc2.w, p.fc2.b, M, C, FF, true));
    // final LN of the last layer writes the CLS rows compactly into h
    TRY(layernorm(cf, cf->x, sx, last ? cf->h : cf->x, last ? C : sx, p.ln2w, p.ln2b, eps, M));
  }
  TRY(gemm(cf, cf->f, P1, cf->h, C, WP(cf, "t.p1.w"), nullptr, B, P1, C, false));
  TRY(gelu(cf, cf->f, (long)B * P1));
  TRY(gemm(cf, cf->emb, E, cf->f, P1, WP(cf, "t.p2.w"), nullptr, B, E, P1, false));
  LAUNCH(cf, k_normalize, (B * 32 + 255) / 256, 256, 0, TP(cf->emb), cf->out, B);
  return 0;
}

// Packed PubMedBERT: cf->ids/pos [T] int32, cf->cu [B+1]. Only real tokens are computed (T may include trailing
// filler tokens that belong to no sequence). Last layer: CLS rows gathered to a compact [B, C] block.
static int text_forward_packed(CF* cf, int B, int NT) {
  const float eps = 1e-12f;
  const int MAXLEN = 256;
  LAUNCH(cf, k_embed_packed, nblk((long)NT * C), 256, 0, cf->ids, cf->pos, TP(WP(cf, "t.word")),
         TP(WP(cf, "t.pos_type")), TP(cf->x), (long)NT);
  TRY(layernorm(cf, cf->x, C, cf->x, C, WP(cf, "t.ln.w"), WP(cf, "t.ln.b"), eps, NT));
  for (int i = 0; i < LAYERS; i++) {
    const Blk& p = cf->tb[i];
    TRY(gemm(cf, cf->qkv, 3 * C, cf->x, C, p.qkv.w, p.qkv.b, NT, 3 * C, C, false));
    if (i < LAYERS - 1) {
      TRY(attention(cf, cf->qkv, cf->a, nullptr, B, MAXLEN, MAXLEN, cf->cu));
      TRY(gemm(cf, cf->x, C, cf->a, C, p.o.w, p.o.b, NT, C, C, true));
      TRY(layernorm(cf, cf->x, C, cf->x, C, p.ln1w, p.ln1b, eps, NT));
      TRY(fc1_gelu(cf, cf->f, cf->x, C, p.fc1.w, p.fc1.b, NT));
      TRY(gemm(cf, cf->x, C, cf->f, FF, p.fc2.w, p.fc2.b, NT, C, FF, true));
      TRY(layernorm(cf, cf->x, C, cf->x, C, p.ln2w, p.ln2b, eps, NT));
    } else {
      TRY(attention(cf, cf->qkv, cf->a, nullptr, B, MAXLEN, 1, cf->cu));
      LAUNCH(cf, k_gather_cls, nblk((long)B * C), 256, 0, TP(cf->a), cf->cu, TP(cf->tmp), B);
      LAUNCH(cf, k_gather_cls, nblk((long)B * C), 256, 0, TP(cf->x), cf->cu, TP(cf->h), B);
      TRY(gemm(cf, cf->h, C, cf->tmp, C, p.o.w, p.o.b, B, C, C, true));
      TRY(layernorm(cf, cf->h, C, cf->h, C, p.ln1w, p.ln1b, eps, B));
      TRY(fc1_gelu(cf, cf->f, cf->h, C, p.fc1.w, p.fc1.b, B));
      TRY(gemm(cf, cf->h, C, cf->f, FF, p.fc2.w, p.fc2.b, B, C, FF, true));
      TRY(layernorm(cf, cf->h, C, cf->h, C, p.ln2w, p.ln2b, eps, B));
    }
  }
  TRY(gemm(cf, cf->f, P1, cf->h, C, WP(cf, "t.p1.w"), nullptr, B, P1, C, false));
  TRY(gelu(cf, cf->f, (long)B * P1));
  TRY(gemm(cf, cf->emb, E, cf->f, P1, WP(cf, "t.p2.w"), nullptr, B, E, P1, false));
  LAUNCH(cf, k_normalize, (B * 32 + 255) / 256, 256, 0, TP(cf->emb), cf->out, B);
  return 0;
}

static int run(CF* cf, int tower, int B, int L, bool graph) {
  auto fwd = [&]() {
    return tower == 0 ? vision_forward(cf, B) : tower == 1 ? text_forward(cf, B, L) : text_forward_packed(cf, B, L);
  };
  if (!graph) return fwd();
  auto key = std::make_tuple(tower, B, L);
  auto it = cf->graphs.find(key);
  if (it == cf->graphs.end()) {
    TRY(fwd());  // eager warm-up: cuBLASLt heuristics + lazy init happen outside capture
    CK(cudaStreamSynchronize(cf->st));
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(cf->st, cudaStreamCaptureModeThreadLocal));
    const int rc = fwd();
    CK(cudaStreamEndCapture(cf->st, &g));
    if (rc) return -1;
    cudaGraphExec_t ex;
    CK(cudaGraphInstantiate(&ex, g, 0));
    cudaGraphDestroy(g);
    it = cf->graphs.emplace(key, ex).first;
  }
  CK(cudaGraphLaunch(it->second, cf->st));
  return 0;
}

extern "C" {

CF* cf_create(int bf16, int max_img, int max_rows) {
  CF* cf = new CF();
  cf->bf16 = bf16 != 0;
  cf->es = cf->bf16 ? 2 : 4;
  cf->max_img = max_img;
  cf->max_rows = max_rows;
  if (cublasLtCreate(&cf->lt) != CUBLAS_STATUS_SUCCESS) return nullptr;
  if (cudaStreamCreateWithFlags(&cf->st, cudaStreamNonBlocking) != cudaSuccess) return nullptr;
  const long rows = std::max<long>((long)max_img * NTOK, max_rows);
  const long maxb = std::max<long>(max_img, max_rows / 8);
  const size_t es = cf->es;
  size_t total = 0;
  auto alloc = [&](void** p, size_t n) { total += n; return cudaMalloc(p, n) == cudaSuccess; };
  bool ok = alloc(&cf->ws, cf->ws_bytes) && alloc(&cf->x, rows * C * es) && alloc(&cf->h, rows * C * es) &&
            alloc(&cf->qkv, rows * 3 * C * es) && alloc(&cf->a, rows * C * es) && alloc(&cf->f, rows * FF * es) &&
            alloc(&cf->tmp, (long)max_img * NPATCH * C * es) &&
            alloc((void**)&cf->px, (long)max_img * 3 * 224 * 224 * 4) && alloc((void**)&cf->ids, rows * 4) && alloc((void**)&cf->pos, rows * 4) &&
            alloc((void**)&cf->cu, (maxb + 1) * 4) &&
            alloc((void**)&cf->lens, maxb * 4) && alloc((void**)&cf->out, maxb * E * 4) && alloc(&cf->emb, maxb * E * es);
  cf->scratch_bytes = 0;  // autotune scratch allocated only when the option is enabled
  cf->act_bytes = total;
  return ok ? cf : nullptr;
}

int cf_set_option(CF* cf, const char* key, int value) {
  if (!strcmp(key, "gelu_epilogue")) { cf->gelu_epilogue = value; return 0; }
  if (!strcmp(key, "autotune")) {
    cf->autotune = value;
    if (value && !cf->scratch) {
      const long rows = std::max<long>((long)cf->max_img * NTOK, cf->max_rows);
      cf->scratch_bytes = (size_t)rows * FF * cf->es;
      CK(cudaMalloc(&cf->scratch, cf->scratch_bytes));
      cf->act_bytes += cf->scratch_bytes;
    }
    return 0;
  }
  if (!strcmp(key, "cutlass_gelu")) { cf->cutlass_gelu = value; return 0; }
  return -1;
}

int cf_set_weight(CF* cf, const char* name, const void* host, size_t bytes) {
  void* d;
  CK(cudaMalloc(&d, bytes));
  CK(cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice));
  cf->W[name] = d;
  cf->weight_bytes += bytes;
  return 0;
}

int cf_finalize(CF* cf) {
  for (int i = 0; i < LAYERS; i++) {
    for (int tw = 0; tw < 2; tw++) {
      Blk& b = tw ? cf->tb[i] : cf->vb[i];
      const std::string p = std::string(tw ? "t." : "v.") + std::to_string(i) + ".";
      void** slots[] = {&b.ln1w, &b.ln1b, &b.ln2w, &b.ln2b, &b.qkv.w, &b.qkv.b, &b.o.w, &b.o.b,
                        &b.fc1.w, &b.fc1.b, &b.fc2.w, &b.fc2.b};
      const char* names[] = {"ln1.w", "ln1.b", "ln2.w", "ln2.b", "qkv.w", "qkv.b", "o.w", "o.b",
                             "fc1.w", "fc1.b", "fc2.w", "fc2.b"};
      for (int k = 0; k < 12; k++) {
        *slots[k] = WP(cf, p + names[k]);
        if (!*slots[k]) return -1;
      }
    }
  }
  for (const char* n : {"v.patch.w", "v.patch.b", "v.cls_pos0", "v.pos", "v.ln.w", "v.ln.b", "v.proj.w", "t.word",
                        "t.pos_type", "t.ln.w", "t.ln.b", "t.p1.w", "t.p2.w"})
    if (!WP(cf, n)) return -1;
  return 0;
}

// Input/output pointers may be host or device memory (cudaMemcpyDefault, UVA), so Python can pass torch tensors.
// flags: 1 = pixels already in the device input buffer (skip H2D), 2 = no CUDA graph, 4 = skip D2H of embeddings
int cf_encode_image(CF* cf, const float* pixels, int n_in, int B, float* out, int n_out, int flags) {
  if (B > cf->max_img || n_in > B || n_out > B) return -2;
  if (!(flags & 1) && n_in > 0)
    CK(cudaMemcpyAsync(cf->px, pixels, (size_t)n_in * 3 * 224 * 224 * 4, cudaMemcpyDefault, cf->st));
  // rows n_in..B of the input buffer keep earlier (finite) data; their embeddings are never copied out
  TRY(run(cf, 0, B, 0, !(flags & 2)));
  if (!(flags & 4) && n_out > 0) CK(cudaMemcpyAsync(out, cf->out, (size_t)n_out * E * 4, cudaMemcpyDefault, cf->st));
  CK(cudaStreamSynchronize(cf->st));
  return 0;
}

// ids [B*L] int32 padded with pad id to L; lens [B] = real length of each row.
int cf_encode_text(CF* cf, const int* ids, const int* lens, int B, int L, float* out, int n_out, int flags) {
  if ((long)B * L > cf->max_rows || L > 512) return -2;
  if (!(flags & 1)) {
    CK(cudaMemcpyAsync(cf->ids, ids, (size_t)B * L * 4, cudaMemcpyDefault, cf->st));
    CK(cudaMemcpyAsync(cf->lens, lens, (size_t)B * 4, cudaMemcpyDefault, cf->st));
  }
  TRY(run(cf, 1, B, L, !(flags & 2)));
  if (!(flags & 4) && n_out > 0) CK(cudaMemcpyAsync(out, cf->out, (size_t)std::min(n_out, B) * E * 4, cudaMemcpyDefault, cf->st));
  CK(cudaStreamSynchronize(cf->st));
  return 0;
}

// Packed varlen text: ids/pos [T] int32 (T may include filler tokens past cu[B]), cu [B+1] int32; each sequence
// at most 256 tokens. One CUDA graph per (B, T).
int cf_encode_text_packed(CF* cf, const int* ids, const int* pos, const int* cu, int B, int T, float* out, int n_out,
                          int flags) {
  if (T > cf->max_rows || B > std::max(cf->max_img, cf->max_rows / 8)) return -2;
  if (!(flags & 1)) {
    CK(cudaMemcpyAsync(cf->ids, ids, (size_t)T * 4, cudaMemcpyDefault, cf->st));
    CK(cudaMemcpyAsync(cf->pos, pos, (size_t)T * 4, cudaMemcpyDefault, cf->st));
    CK(cudaMemcpyAsync(cf->cu, cu, (size_t)(B + 1) * 4, cudaMemcpyDefault, cf->st));
  }
  TRY(run(cf, 2, B, T, !(flags & 2)));
  if (!(flags & 4) && n_out > 0) CK(cudaMemcpyAsync(out, cf->out, (size_t)std::min(n_out, B) * E * 4, cudaMemcpyDefault, cf->st));
  CK(cudaStreamSynchronize(cf->st));
  return 0;
}

int cf_mem(CF* cf, size_t* weight_bytes, size_t* act_bytes, size_t* dev_used, size_t* dev_total) {
  size_t fr, tot;
  CK(cudaMemGetInfo(&fr, &tot));
  *weight_bytes = cf->weight_bytes;
  *act_bytes = cf->act_bytes;
  *dev_used = tot - fr;
  *dev_total = tot;
  return 0;
}

int cf_device_name(char* buf, int n) {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  snprintf(buf, n, "%s", p.name);
  return 0;
}

void cf_destroy(CF* cf) {
  if (!cf) return;
  for (auto& kv : cf->graphs) cudaGraphExecDestroy(kv.second);
  for (auto& kv : cf->W) cudaFree(kv.second);
  for (void* p : {cf->scratch, cf->ws, cf->x, cf->h, cf->qkv, cf->a, cf->f, cf->tmp, (void*)cf->px, (void*)cf->ids,
                  (void*)cf->pos, (void*)cf->cu,
                  (void*)cf->lens, (void*)cf->out, cf->emb})
    cudaFree(p);
  cublasLtDestroy(cf->lt);
  cudaStreamDestroy(cf->st);
  delete cf;
}

}  // extern "C"
