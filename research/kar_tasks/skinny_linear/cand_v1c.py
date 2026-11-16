"""Skinny linear y = x @ w.T + b (BF16 in/out, FP32 accumulate) for M <= 64 on sm_120.

Design: weight-streaming split-K mma.sync.m16n8k16. Each warp owns 8 output columns (one n8 tile) and a K slice;
every lane loads 16 contiguous bytes of one weight row per 32-k chunk (fully coalesced, read once), and the k order
inside a chunk is permuted identically for A and B so fragments come straight from those 16-byte loads. Warps of a
CTA split K; partial sums meet in shared memory, where bias is added and BF16 written. Other shapes: cuBLAS.
Tunables (module constants) are compile-time template arguments.
"""
import ctypes
import struct

import torch
import torch.nn.functional as F
from cuda.bindings import driver, nvrtc

WN, WK, CH = 2, 4, 4          # n8 tiles per CTA, K-split warps per CTA, 32-k chunks loaded per batch

SRC = r"""
typedef unsigned short bf16;
typedef unsigned long long u64;
struct Params { u64 x, w, b, y; long long M, N, K; };

__device__ __forceinline__ float bf2f(bf16 v) { return __uint_as_float(((unsigned)v) << 16); }
__device__ __forceinline__ bf16 f2bf(float f) {
  unsigned u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (bf16)(u >> 16);
}
__device__ __forceinline__ uint4 ldw(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void mma(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                    unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int MT, int WN_, int WK_, int CH_>
__device__ __forceinline__ void body(const Params& p) {
  const bf16* X = reinterpret_cast<const bf16*>(p.x);
  const bf16* W = reinterpret_cast<const bf16*>(p.w);
  const bf16* Bv = reinterpret_cast<const bf16*>(p.b);
  bf16* Y = reinterpret_cast<bf16*>(p.y);
  const int M = (int)p.M, N = (int)p.N, K = (int)p.K;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int wn = warp % WN_, wk = warp / WN_;
  const int n0 = (blockIdx.x * WN_ + wn) * 8;
  const int Kw = K / WK_, kbeg = wk * Kw, kend = kbeg + Kw;
  float acc[MT][4];
#pragma unroll
  for (int i = 0; i < MT; i++) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
  const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)(n0 + g) * K);
  const uint4 z = make_uint4(0, 0, 0, 0);
  for (int kc = kbeg; kc < kend; kc += 32 * CH_) {
    uint4 wv[CH_];
#pragma unroll
    for (int c = 0; c < CH_; c++) {
      const int k = kc + c * 32 + t * 8;
      wv[c] = (kc + c * 32 < kend) ? ldw(wrow + (k >> 3)) : z;
    }
#pragma unroll
    for (int c = 0; c < CH_; c++) {
      if (kc + c * 32 >= kend) break;
      const int k = kc + c * 32 + t * 8;
#pragma unroll
      for (int mt = 0; mt < MT; mt++) {
        const int r0 = mt * 16 + g, r1 = r0 + 8;
        const uint4 xa = r0 < M ? *reinterpret_cast<const uint4*>(X + (size_t)r0 * K + k) : z;
        const uint4 xb = r1 < M ? *reinterpret_cast<const uint4*>(X + (size_t)r1 * K + k) : z;
        mma(acc[mt], xa.x, xb.x, xa.y, xb.y, wv[c].x, wv[c].y);
        mma(acc[mt], xa.z, xb.z, xa.w, xb.w, wv[c].z, wv[c].w);
      }
    }
  }
  __shared__ float red[WK_ * WN_ * MT * 16 * 8];
  float* R = red + ((wk * WN_ + wn) * MT * 16) * 8;
#pragma unroll
  for (int mt = 0; mt < MT; mt++) {
    R[(mt * 16 + g) * 8 + t * 2 + 0] = acc[mt][0];
    R[(mt * 16 + g) * 8 + t * 2 + 1] = acc[mt][1];
    R[(mt * 16 + g + 8) * 8 + t * 2 + 0] = acc[mt][2];
    R[(mt * 16 + g + 8) * 8 + t * 2 + 1] = acc[mt][3];
  }
  __syncthreads();
  const int per = MT * 16 * WN_ * 8;
  for (int i = threadIdx.x; i < per; i += blockDim.x) {
    const int row = i / (WN_ * 8), cc = i % (WN_ * 8), w2 = cc >> 3, c = cc & 7;
    if (row >= M) continue;
    float s = 0.f;
#pragma unroll
    for (int q = 0; q < WK_; q++) s += red[((q * WN_ + w2) * MT * 16 + row) * 8 + c];
    const int n = blockIdx.x * WN_ * 8 + cc;
    Y[(size_t)row * N + n] = f2bf(s + bf2f(Bv[n]));
  }
}

extern "C" __global__ void skinny_mt1(const Params p) { body<1, WN_C, WK_C, CH_C>(p); }
extern "C" __global__ void skinny_mt2(const Params p) { body<2, WN_C, WK_C, CH_C>(p); }
extern "C" __global__ void skinny_mt3(const Params p) { body<3, WN_C, WK_C, CH_C>(p); }
extern "C" __global__ void skinny_mt4(const Params p) { body<4, WN_C, WK_C, CH_C>(p); }
"""


def _check(err, what):
    if int(err):
        raise RuntimeError(f"{what}: {err}")


class _Module:
    def __init__(self, src, names, defines):
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"skinny.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"-std=c++17"]
        opts += [f"-D{k}={v}".encode() for k, v in defines.items()]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        _, n = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        image = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, image)
        err, dev = driver.cuDeviceGet(torch.cuda.current_device())
        _check(err, "cuDeviceGet")
        err, self.ctx = driver.cuDevicePrimaryCtxRetain(dev)
        _check(err, "cuDevicePrimaryCtxRetain")
        err, self.mod = driver.cuModuleLoadData(image)
        _check(err, "cuModuleLoadData")
        self.fns = {}
        for nm in names:
            err, f = driver.cuModuleGetFunction(self.mod, nm.encode())
            _check(err, "cuModuleGetFunction " + nm)
            self.fns[nm] = f

    def launch(self, name, grid, block, params):
        err, cur = driver.cuCtxGetCurrent()
        if int(cur) != int(self.ctx):
            (err,) = driver.cuCtxSetCurrent(self.ctx)
            _check(err, "cuCtxSetCurrent")
        buf = ctypes.create_string_buffer(params, len(params))
        argv = (ctypes.c_void_p * 1)(ctypes.addressof(buf))
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = driver.cuLaunchKernel(self.fns[name], grid, 1, 1, block, 1, 1, 0, stream, ctypes.addressof(argv), 0)
        _check(err, "cuLaunchKernel")


_M = _Module(SRC, [f"skinny_mt{i}" for i in (1, 2, 3, 4)], {"WN_C": WN, "WK_C": WK, "CH_C": CH})


def kernel_fn(x, w, b):
    M, K = x.shape
    N = w.shape[0]
    ok = (M <= 64 and N % (8 * WN) == 0 and K % (32 * WK) == 0 and x.is_contiguous() and w.is_contiguous()
          and x.dtype == w.dtype == b.dtype == torch.bfloat16 and b.is_contiguous()
          and x.data_ptr() % 16 == 0 and w.data_ptr() % 16 == 0)
    if not ok:
        return F.linear(x, w, b)
    y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    mt = (M + 15) // 16
    params = struct.pack("<4Q3q", x.data_ptr(), w.data_ptr(), b.data_ptr(), y.data_ptr(), M, N, K)
    _M.launch(f"skinny_mt{mt}", N // (8 * WN), 32 * WN * WK, params)
    return y
