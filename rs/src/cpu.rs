//! CPU inference engine for BiomedCLIP (x86-64 with AVX-512; tuned for Zen 5). Linear layers in one of three formats:
//!   F32   - exact parity mode (packed weight panels, AVX-512 FMA)
//!   Bf16  - bf16 weights and activations, f32 accumulate (AVX-512 BF16 `vdpbf16ps`)
//!   Q8Blk - block-scaled W8A8: int8 weights and activations with one f32 scale per 32 K-values (per output row for
//!           weights, per input row for activations), int32 accumulate with AVX-512 VNNI `vpdpbusd`
//! LayerNorm, GELU, attention and the residual stream are f32. Same math and folding as the GPU engine (CLS-only
//! last layer, packed text without padding).
use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;
use safetensors::SafeTensors;
use std::collections::HashMap;

const C: usize = 768;
const H: usize = 12;
const D: usize = 64;
const FF: usize = 3072;
const E: usize = 512;
const P1: usize = 640;
const NTOK: usize = 197;
const NPATCH: usize = 196;
const LAYERS: usize = 12;
const NT: usize = 16; // output columns per panel (one zmm of f32)
const QB: usize = 32; // Q8Blk block size along K

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Format {
    F32,
    Bf16,
    Q8Blk,
}

/// A linear layer y = x W^T + b, weights pre-packed into 16-column panels.
struct Lin {
    n: usize,
    k: usize,
    bias: Option<Vec<f32>>,
    f32p: Vec<f32>,        // [n/16][k][16]
    bf16p: Vec<u32>,       // [n/16][k/2][16] pairs (lo = k even)
    q8p: Vec<i8>,          // [n/16][k/4][16][4]
    q8s: Vec<f32>,         // [n/16][k/32][16] weight block scales
    q8sum: Vec<i32>,       // [n/16][k/32][16] 128 * sum of the block's int8 weights (unsigned-shift correction)
}

fn bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 {
        return ((u >> 16) | 0x40) as u16;
    }
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

impl Lin {
    fn new(w: &[f32], n: usize, k: usize, bias: Option<Vec<f32>>, fmt: Format) -> Self {
        assert!(n % (2 * NT) == 0 && k % QB == 0, "CPU linears need N % 32 == 0 and K % 32 == 0");
        let mut l = Lin { n, k, bias, f32p: vec![], bf16p: vec![], q8p: vec![], q8s: vec![], q8sum: vec![] };
        let np = n / NT;
        match fmt {
            Format::F32 => {
                l.f32p = vec![0f32; n * k];
                for p in 0..np {
                    for kk in 0..k {
                        for j in 0..NT {
                            l.f32p[(p * k + kk) * NT + j] = w[(p * NT + j) * k + kk];
                        }
                    }
                }
            }
            Format::Bf16 => {
                l.bf16p = vec![0u32; n * k / 2];
                for p in 0..np {
                    for k2 in 0..k / 2 {
                        for j in 0..NT {
                            let r = (p * NT + j) * k;
                            l.bf16p[(p * (k / 2) + k2) * NT + j] =
                                bf16_bits(w[r + 2 * k2]) as u32 | ((bf16_bits(w[r + 2 * k2 + 1]) as u32) << 16);
                        }
                    }
                }
            }
            Format::Q8Blk => {
                let nb = k / QB;
                l.q8p = vec![0i8; n * k];
                l.q8s = vec![0f32; n * nb];
                l.q8sum = vec![0i32; n * nb];
                for row in 0..n {
                    let (p, j) = (row / NT, row % NT);
                    for b in 0..nb {
                        let blk = &w[row * k + b * QB..row * k + (b + 1) * QB];
                        let amax = blk.iter().fold(0f32, |m, x| m.max(x.abs()));
                        let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                        let mut sum = 0i32;
                        for (t, x) in blk.iter().enumerate() {
                            let q = (x / s).round().clamp(-127.0, 127.0) as i8;
                            sum += q as i32;
                            let kk = b * QB + t;
                            l.q8p[((p * (k / 4) + kk / 4) * NT + j) * 4 + kk % 4] = q;
                        }
                        l.q8s[(p * nb + b) * NT + j] = s;
                        l.q8sum[(p * nb + b) * NT + j] = 128 * sum;
                    }
                }
            }
        }
        l
    }

    fn bytes(&self) -> usize {
        self.f32p.len() * 4 + self.bf16p.len() * 4 + self.q8p.len() + self.q8s.len() * 4 + self.q8sum.len() * 4
    }
}

// ---------------------------------------------------------------- kernels (AVX-512)
#[cfg(target_arch = "x86_64")]
mod k {
    use super::*;
    use std::arch::x86_64::*;

    /// MR rows x 32 columns (panels p, p+1): f32
    #[target_feature(enable = "avx512f")]
    pub unsafe fn f32_tile<const MR: usize>(a: &[f32], lda: usize, m0: usize, w: &[f32], k: usize, p: usize, out: &mut [[__m512; 2]; MR]) {
        let mut acc = [[_mm512_setzero_ps(); 2]; MR];
        let w0 = w.as_ptr().add(p * k * NT);
        let w1 = w.as_ptr().add((p + 1) * k * NT);
        let ap = a.as_ptr().add(m0 * lda);
        for kk in 0..k {
            let v0 = _mm512_loadu_ps(w0.add(kk * NT));
            let v1 = _mm512_loadu_ps(w1.add(kk * NT));
            for i in 0..MR {
                let av = _mm512_set1_ps(*ap.add(i * lda + kk));
                acc[i][0] = _mm512_fmadd_ps(av, v0, acc[i][0]);
                acc[i][1] = _mm512_fmadd_ps(av, v1, acc[i][1]);
            }
        }
        *out = acc;
    }

    /// MR rows x 32 columns: bf16 pairs (vdpbf16ps)
    #[target_feature(enable = "avx512f,avx512bf16")]
    pub unsafe fn bf16_tile<const MR: usize>(a2: &[u32], k2: usize, m0: usize, w: &[u32], p: usize, out: &mut [[__m512; 2]; MR]) {
        let mut acc = [[_mm512_setzero_ps(); 2]; MR];
        let w0 = w.as_ptr().add(p * k2 * NT);
        let w1 = w.as_ptr().add((p + 1) * k2 * NT);
        let ap = a2.as_ptr().add(m0 * k2);
        for kk in 0..k2 {
            let v0: __m512bh = std::mem::transmute(_mm512_loadu_si512(w0.add(kk * NT) as *const _));
            let v1: __m512bh = std::mem::transmute(_mm512_loadu_si512(w1.add(kk * NT) as *const _));
            for i in 0..MR {
                let av: __m512bh = std::mem::transmute(_mm512_set1_epi32(*ap.add(i * k2 + kk) as i32));
                acc[i][0] = _mm512_dpbf16_ps(acc[i][0], av, v0);
                acc[i][1] = _mm512_dpbf16_ps(acc[i][1], av, v1);
            }
        }
        *out = acc;
    }

    /// MR rows x 32 columns: Q8Blk (vpdpbusd per 32-K block, then scaled f32 accumulate)
    #[target_feature(enable = "avx512f,avx512vnni")]
    pub unsafe fn q8_tile<const MR: usize>(a8u: &[u8], sa: &[f32], m0: usize, l: &Lin, p: usize, out: &mut [[__m512; 2]; MR]) {
        let (k, nb) = (l.k, l.k / QB);
        let mut acc = [[_mm512_setzero_ps(); 2]; MR];
        let w0 = l.q8p.as_ptr().add(p * (k / 4) * NT * 4);
        let w1 = l.q8p.as_ptr().add((p + 1) * (k / 4) * NT * 4);
        let ap = a8u.as_ptr().add(m0 * k);
        for b in 0..nb {
            let mut ia = [[_mm512_setzero_si512(); 2]; MR];
            for kk in 0..QB / 4 {
                let k4 = b * (QB / 4) + kk;
                let v0 = _mm512_loadu_si512(w0.add(k4 * NT * 4) as *const _);
                let v1 = _mm512_loadu_si512(w1.add(k4 * NT * 4) as *const _);
                for i in 0..MR {
                    let av = _mm512_set1_epi32((ap.add(i * k + k4 * 4) as *const i32).read_unaligned());
                    ia[i][0] = _mm512_dpbusd_epi32(ia[i][0], av, v0);
                    ia[i][1] = _mm512_dpbusd_epi32(ia[i][1], av, v1);
                }
            }
            let s0 = _mm512_loadu_ps(l.q8s.as_ptr().add((p * nb + b) * NT));
            let s1 = _mm512_loadu_ps(l.q8s.as_ptr().add(((p + 1) * nb + b) * NT));
            let c0 = _mm512_loadu_si512(l.q8sum.as_ptr().add((p * nb + b) * NT) as *const _);
            let c1 = _mm512_loadu_si512(l.q8sum.as_ptr().add(((p + 1) * nb + b) * NT) as *const _);
            for i in 0..MR {
                let sav = _mm512_set1_ps(*sa.get_unchecked((m0 + i) * nb + b));
                acc[i][0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(ia[i][0], c0)), _mm512_mul_ps(s0, sav), acc[i][0]);
                acc[i][1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(ia[i][1], c1)), _mm512_mul_ps(s1, sav), acc[i][1]);
            }
        }
        *out = acc;
    }

    /// 32 floats at a time -> bf16 pairs (round to nearest even), as bf16_bits
    #[target_feature(enable = "avx512f,avx512bf16")]
    pub unsafe fn to_bf16_pairs(x: &[f32], out: &mut [u32]) {
        for c in 0..x.len() / 32 {
            let lo = _mm512_loadu_ps(x.as_ptr().add(c * 32));
            let hi = _mm512_loadu_ps(x.as_ptr().add(c * 32 + 16));
            let v: __m512i = std::mem::transmute(_mm512_cvtne2ps_pbh(hi, lo));
            _mm512_storeu_si512(out.as_mut_ptr().add(c * 16) as *mut _, v);
        }
    }

    /// per 32-value block: scale = max|x| / 127, q = round(x / scale) + 128 as u8
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn quant_row_u8(x: &[f32], out: &mut [u8], scales: &mut [f32]) {
        let sign = _mm512_set1_ps(-0.0);
        for b in 0..x.len() / 32 {
            let v0 = _mm512_loadu_ps(x.as_ptr().add(b * 32));
            let v1 = _mm512_loadu_ps(x.as_ptr().add(b * 32 + 16));
            let m = _mm512_max_ps(_mm512_andnot_ps(sign, v0), _mm512_andnot_ps(sign, v1));
            let amax = _mm512_reduce_max_ps(m);
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            scales[b] = s;
            let inv = _mm512_set1_ps(1.0 / s);
            let lim = _mm512_set1_ps(127.0);
            let q0 = _mm512_cvtps_epi32(_mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(v0, inv), _mm512_set1_ps(-127.0)), lim));
            let q1 = _mm512_cvtps_epi32(_mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(v1, inv), _mm512_set1_ps(-127.0)), lim));
            let off = _mm512_set1_epi32(128);
            let u0 = _mm512_cvtepi32_epi8(_mm512_add_epi32(q0, off));
            let u1 = _mm512_cvtepi32_epi8(_mm512_add_epi32(q1, off));
            _mm_storeu_si128(out.as_mut_ptr().add(b * 32) as *mut _, u0);
            _mm_storeu_si128(out.as_mut_ptr().add(b * 32 + 16) as *mut _, u1);
        }
    }

    /// 16-wide dot product (f32), AVX-512
    #[target_feature(enable = "avx512f")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = _mm512_setzero_ps();
        let n = a.len() / 16 * 16;
        for i in (0..n).step_by(16) {
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i)), _mm512_loadu_ps(b.as_ptr().add(i)), acc);
        }
        let mut s = _mm512_reduce_add_ps(acc);
        for i in n..a.len() {
            s += a[i] * b[i];
        }
        s
    }

    /// o += p * v (16-wide)
    #[target_feature(enable = "avx512f")]
    pub unsafe fn axpy(p: f32, v: &[f32], o: &mut [f32]) {
        let pv = _mm512_set1_ps(p);
        for i in (0..v.len()).step_by(16) {
            let r = _mm512_fmadd_ps(pv, _mm512_loadu_ps(v.as_ptr().add(i)), _mm512_loadu_ps(o.as_ptr().add(i)));
            _mm512_storeu_ps(o.as_mut_ptr().add(i), r);
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn store(v: __m512, dst: *mut f32) {
        _mm512_storeu_ps(dst, v);
    }
}

/// Epilogue of a linear: 0 = bias, 1 = bias + exact GELU, 2 = bias + residual add (y += ...)
#[derive(Clone, Copy, PartialEq)]
enum Epi {
    Bias,
    Gelu,
    Residual,
}

/// erf with |error| <= 3e-7 (Abramowitz & Stegun 7.1.28): no exp, so loops over slices auto-vectorize.
#[inline(always)]
fn erf_f32(x: f32) -> f32 {
    let a = x.abs();
    let t = 1.0 + a * (0.0705230784 + a * (0.0422820123 + a * (0.0092705272 + a * (0.0001520143 + a * (0.0002765672 + a * 0.0000430638)))));
    let t2 = t * t;
    let t4 = t2 * t2;
    let t8 = t4 * t4;
    let r = 1.0 - 1.0 / (t8 * t8);
    if x < 0.0 { -r } else { r }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gelu_slice(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = 0.5 * *x * (1.0 + erf_f32(*x * std::f32::consts::FRAC_1_SQRT_2));
    }
}

pub struct CpuEngine {
    fmt: Format,
    pool: rayon::ThreadPool,
    f: HashMap<String, Vec<f32>>, // embeddings, norms, small tensors
    v: Vec<[Lin; 4]>,             // qkv, o, fc1, fc2 per layer (vision)
    t: Vec<[Lin; 4]>,             // (text)
    v_patch: Lin,
    v_proj: Lin,
    t_p1: Lin,
    t_p2: Lin,
    pub weight_bytes: usize,
}

impl CpuEngine {
    pub fn new(assets: &str, fmt: Format, threads: Option<usize>, pin: bool) -> Result<Self> {
        if !is_x86_feature_detected!("avx512f") {
            bail!("CPU engine needs AVX-512");
        }
        if fmt == Format::Bf16 && !is_x86_feature_detected!("avx512bf16") {
            bail!("bf16 format needs AVX-512 BF16");
        }
        if fmt == Format::Q8Blk && !is_x86_feature_detected!("avx512vnni") {
            bail!("q8blk format needs AVX-512 VNNI");
        }
        let n = threads.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(16));
        let cpus: Vec<usize> = affinity();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .start_handler(move |i| {
                if pin && !cpus.is_empty() {
                    pin_to(cpus[i % cpus.len()]);
                }
            })
            .build()?;
        let bytes = std::fs::read(format!("{assets}/weights_f32.safetensors"))?;
        let st = SafeTensors::deserialize(&bytes)?;
        let get = |name: &str| -> Result<Vec<f32>> {
            let t = st.tensor(name).map_err(|e| anyhow!("{name}: {e}"))?;
            Ok(t.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        };
        let mut f = HashMap::new();
        for nm in ["v.cls_pos0", "v.pos", "v.ln.w", "v.ln.b", "t.word", "t.pos_type", "t.ln.w", "t.ln.b"] {
            f.insert(nm.to_string(), get(nm)?);
        }
        let mut wb = 0usize;
        let mut lin = |w: &str, b: Option<&str>, n: usize, k: usize| -> Result<Lin> {
            let l = Lin::new(&get(w)?, n, k, b.map(|b| get(b)).transpose()?, fmt);
            wb += l.bytes() + l.bias.as_ref().map_or(0, |v| v.len() * 4);
            Ok(l)
        };
        let v_patch = lin("v.patch.w", Some("v.patch.b"), C, C)?;
        let v_proj = lin("v.proj.w", None, E, C)?;
        let t_p1 = lin("t.p1.w", None, P1, C)?;
        let t_p2 = lin("t.p2.w", None, E, P1)?;
        let mut v = vec![];
        let mut t = vec![];
        for tw in ["v", "t"] {
            for i in 0..LAYERS {
                let p = format!("{tw}.{i}.");
                let blk = [lin(&(p.clone() + "qkv.w"), Some(&(p.clone() + "qkv.b")), 3 * C, C)?,
                           lin(&(p.clone() + "o.w"), Some(&(p.clone() + "o.b")), C, C)?,
                           lin(&(p.clone() + "fc1.w"), Some(&(p.clone() + "fc1.b")), FF, C)?,
                           lin(&(p.clone() + "fc2.w"), Some(&(p.clone() + "fc2.b")), C, FF)?];
                for nm in ["ln1.w", "ln1.b", "ln2.w", "ln2.b"] {
                    f.insert(format!("{p}{nm}"), get(&(p.clone() + nm))?);
                }
                if tw == "v" { v.push(blk) } else { t.push(blk) }
            }
        }
        for vv in f.values() {
            wb += vv.len() * 4;
        }
        Ok(CpuEngine { fmt, pool, f, v, t, v_patch, v_proj, t_p1, t_p2, weight_bytes: wb })
    }

    pub fn threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// y[M, N] (row stride ldy) = epi(x[M, K] (row stride ldx) W^T + b)
    fn linear(&self, x: &[f32], ldx: usize, m: usize, l: &Lin, y: &mut [f32], ldy: usize, epi: Epi) {
        let (n, k) = (l.n, l.k);
        const MR: usize = 8;
        const MB: usize = 32; // rows per task (4 register tiles): the weight panel pair stays in L1
        let mp = m.div_ceil(MB) * MB; // rows padded with zeros so every tile is in bounds
        // activation preprocessing per format (contiguous, padded copies)
        let (xp, a2, a8u, sa): (Vec<f32>, Vec<u32>, Vec<u8>, Vec<f32>) = match self.fmt {
            Format::F32 => {
                let mut xp = vec![0f32; mp * k];
                xp.par_chunks_mut(k).take(m).enumerate().for_each(|(r, row)| row.copy_from_slice(&x[r * ldx..r * ldx + k]));
                (xp, vec![], vec![], vec![])
            }
            Format::Bf16 => {
                let mut a2 = vec![0u32; mp * k / 2];
                a2.par_chunks_mut(k / 2).take(m).enumerate().for_each(|(r, row)| unsafe {
                    k::to_bf16_pairs(&x[r * ldx..r * ldx + k], row)
                });
                (vec![], a2, vec![], vec![])
            }
            Format::Q8Blk => {
                let nb = k / QB;
                let mut a8u = vec![128u8; mp * k];
                let mut sa = vec![0f32; mp * nb];
                a8u.par_chunks_mut(k).zip(sa.par_chunks_mut(nb)).take(m).enumerate().for_each(|(r, (row, srow))| unsafe {
                    k::quant_row_u8(&x[r * ldx..r * ldx + k], row, srow)
                });
                (vec![], vec![], a8u, sa)
            }
        };
        let np2 = n / (2 * NT);
        let mblocks = m.div_ceil(MB);
        let yptr = y.as_mut_ptr() as usize;
        let fmt = self.fmt;
        // parallel over (32-row block, panel pair); each task writes a disjoint block of y
        (0..mblocks * np2).into_par_iter().for_each(|task| {
            let (mb, pp) = (task / np2, task % np2);
            let p = 2 * pp;
            for m0 in (mb * MB..((mb + 1) * MB).min(m)).step_by(MR) {
                let rows = MR.min(m - m0);
                let mut acc = unsafe { [[std::mem::zeroed::<std::arch::x86_64::__m512>(); 2]; MR] };
                unsafe {
                    match fmt {
                        Format::F32 => k::f32_tile::<MR>(&xp, k, m0, &l.f32p, k, p, &mut acc),
                        Format::Bf16 => k::bf16_tile::<MR>(&a2, k / 2, m0, &l.bf16p, p, &mut acc),
                        Format::Q8Blk => k::q8_tile::<MR>(&a8u, &sa, m0, l, p, &mut acc),
                    }
                }
                let mut tile = [[0f32; 2 * NT]; MR];
                for i in 0..MR {
                    unsafe {
                        k::store(acc[i][0], tile[i].as_mut_ptr());
                        k::store(acc[i][1], tile[i].as_mut_ptr().add(NT));
                    }
                }
                for i in 0..rows {
                    let r = m0 + i;
                    let dst = unsafe { std::slice::from_raw_parts_mut((yptr as *mut f32).add(r * ldy + p * NT), 2 * NT) };
                    let row = &mut tile[i];
                    if let Some(b) = l.bias.as_ref() {
                        for j in 0..2 * NT {
                            row[j] += b[p * NT + j];
                        }
                    }
                    match epi {
                        Epi::Bias => dst.copy_from_slice(row),
                        Epi::Gelu => {
                            unsafe { gelu_slice(row) };
                            dst.copy_from_slice(row);
                        }
                        Epi::Residual => {
                            for j in 0..2 * NT {
                                dst[j] += row[j];
                            }
                        }
                    }
                }
            }
        });
    }

    fn layernorm(x: &mut [f32], ld: usize, rows: usize, w: &[f32], b: &[f32], eps: f32, out: Option<(&mut [f32], usize)>) {
        let norm = |src: &[f32], dst: &mut [f32]| {
            let ones = [1f32; C];
            let mean = unsafe { k::dot(src, &ones) } / C as f32;
            let cen: Vec<f32> = src.iter().map(|v| v - mean).collect();
            let var = unsafe { k::dot(&cen, &cen) } / C as f32;
            let r = 1.0 / (var + eps).sqrt();
            for c in 0..C {
                dst[c] = (src[c] - mean) * r * w[c] + b[c];
            }
        };
        match out {
            None => x.par_chunks_mut(ld).take(rows).for_each(|row| {
                let src = row[..C].to_vec();
                norm(&src, &mut row[..C]);
            }),
            Some((o, ldo)) => o.par_chunks_mut(ldo).take(rows).enumerate().for_each(|(r, dst)| {
                norm(&x[r * ld..r * ld + C], &mut dst[..C]);
            }),
        }
    }

    /// attention for sequences given by (start row, length); queries: all rows or only the first (CLS).
    /// Per (sequence, head): K is transposed once so scores are 16-key-wide FMAs; P*V accumulates rows of V.
    fn attention(qkv: &[f32], seqs: &[(usize, usize)], cls_only: bool, out: &mut [f32]) {
        let optr = out.as_mut_ptr() as usize;
        seqs.par_iter().flat_map(|&s| (0..H).into_par_iter().map(move |h| (s, h))).for_each(|((r0, len), h)| {
            let nq = if cls_only { 1 } else { len };
            let lp = len.div_ceil(16) * 16;
            let mut kt = vec![0f32; D * lp];
            for j in 0..len {
                let kk = &qkv[(r0 + j) * 3 * C + C + h * D..(r0 + j) * 3 * C + C + h * D + D];
                for t in 0..D {
                    kt[t * lp + j] = kk[t];
                }
            }
            let mut sc = vec![0f32; lp];
            for qi in 0..nq {
                let q = &qkv[(r0 + qi) * 3 * C + h * D..(r0 + qi) * 3 * C + h * D + D];
                sc.iter_mut().for_each(|v| *v = 0.0);
                for t in 0..D {
                    unsafe { k::axpy(q[t] * 0.125, &kt[t * lp..(t + 1) * lp], &mut sc) };
                }
                let mx = sc[..len].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut den = 0f32;
                for v in sc[..len].iter_mut() {
                    *v = (*v - mx).exp();
                    den += *v;
                }
                let inv = 1.0 / den;
                let mut o = [0f32; D];
                for j in 0..len {
                    let vv = &qkv[(r0 + j) * 3 * C + 2 * C + h * D..(r0 + j) * 3 * C + 2 * C + h * D + D];
                    unsafe { k::axpy(sc[j] * inv, vv, &mut o) };
                }
                let dst = unsafe { std::slice::from_raw_parts_mut((optr as *mut f32).add((r0 + qi) * C + h * D), D) };
                dst.copy_from_slice(&o);
            }
        });
    }

    fn normalize(v: &mut [f32]) {
        for row in v.chunks_mut(E) {
            let n = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            row.iter_mut().for_each(|x| *x /= n);
        }
    }

    /// pixels: B * 3*224*224 open_clip-preprocessed -> B x 512 L2-normalized
    pub fn encode_images(&self, px: &[f32]) -> Vec<f32> {
        self.pool.install(|| {
            let b = px.len() / (3 * 224 * 224);
            let r = b * NTOK;
            // patchify (c, ky, kx) -> GEMM
            let mut patches = vec![0f32; b * NPATCH * C];
            patches.par_chunks_mut(C).enumerate().for_each(|(row, dst)| {
                let (bi, pp) = (row / NPATCH, row % NPATCH);
                let (py, pxx) = (pp / 14, pp % 14);
                for c in 0..3 {
                    for ky in 0..16 {
                        for kx in 0..16 {
                            dst[c * 256 + ky * 16 + kx] = px[((bi * 3 + c) * 224 + py * 16 + ky) * 224 + pxx * 16 + kx];
                        }
                    }
                }
            });
            let mut tmp = vec![0f32; b * NPATCH * C];
            self.linear(&patches, C, b * NPATCH, &self.v_patch, &mut tmp, C, Epi::Bias);
            let (cls, pos) = (&self.f["v.cls_pos0"], &self.f["v.pos"]);
            let mut x = vec![0f32; r * C];
            x.par_chunks_mut(C).enumerate().for_each(|(row, dst)| {
                let (bi, t) = (row / NTOK, row % NTOK);
                for c in 0..C {
                    dst[c] = if t == 0 { cls[c] } else { tmp[(bi * NPATCH + t - 1) * C + c] + pos[(t - 1) * C + c] };
                }
            });
            let seqs: Vec<(usize, usize)> = (0..b).map(|i| (i * NTOK, NTOK)).collect();
            let (mut h, mut qkv, mut a, mut f) = (vec![0f32; r * C], vec![0f32; r * 3 * C], vec![0f32; r * C], vec![0f32; r * FF]);
            for i in 0..LAYERS {
                let p = format!("v.{i}.");
                let l = &self.v[i];
                let last = i == LAYERS - 1;
                Self::layernorm(&mut x, C, r, &self.f[&(p.clone() + "ln1.w")], &self.f[&(p.clone() + "ln1.b")], 1e-6, Some((&mut h, C)));
                self.linear(&h, C, r, &l[0], &mut qkv, 3 * C, Epi::Bias);
                Self::attention(&qkv, &seqs, last, &mut a);
                // last layer: CLS rows only (row stride NTOK*C)
                let (m, s) = if last { (b, NTOK * C) } else { (r, C) };
                self.linear(&a, s, m, &l[1], &mut x, s, Epi::Residual);
                Self::layernorm(&mut x, s, m, &self.f[&(p.clone() + "ln2.w")], &self.f[&(p.clone() + "ln2.b")], 1e-6, Some((&mut h, C)));
                self.linear(&h, C, m, &l[2], &mut f, FF, Epi::Gelu);
                self.linear(&f, FF, m, &l[3], &mut x, s, Epi::Residual);
            }
            Self::layernorm(&mut x, NTOK * C, b, &self.f["v.ln.w"], &self.f["v.ln.b"], 1e-6, Some((&mut h, C)));
            let mut emb = vec![0f32; b * E];
            self.linear(&h, C, b, &self.v_proj, &mut emb, E, Epi::Bias);
            Self::normalize(&mut emb);
            emb
        })
    }

    /// token rows ([CLS] ... [SEP], <= 256 tokens) -> B x 512 L2-normalized (packed: no padding)
    pub fn encode_texts(&self, rows: &[Vec<i32>]) -> Vec<f32> {
        self.pool.install(|| {
            let b = rows.len();
            let mut seqs = Vec::with_capacity(b);
            let mut t = 0usize;
            for r in rows {
                let n = r.len().min(256);
                seqs.push((t, n));
                t += n;
            }
            let (word, pt) = (&self.f["t.word"], &self.f["t.pos_type"]);
            let mut x = vec![0f32; t * C];
            let mut tok = Vec::with_capacity(t);
            for r in rows {
                for (pos, &id) in r.iter().take(256).enumerate() {
                    tok.push((id as usize, pos));
                }
            }
            x.par_chunks_mut(C).zip(tok.par_iter()).for_each(|(dst, &(id, pos))| {
                for c in 0..C {
                    dst[c] = word[id * C + c] + pt[pos * C + c];
                }
            });
            Self::layernorm(&mut x, C, t, &self.f["t.ln.w"], &self.f["t.ln.b"], 1e-12, None);
            let (mut qkv, mut a, mut f) = (vec![0f32; t * 3 * C], vec![0f32; t * C], vec![0f32; t * FF]);
            let mut xc = vec![0f32; b * C];
            for i in 0..LAYERS {
                let p = format!("t.{i}.");
                let l = &self.t[i];
                let last = i == LAYERS - 1;
                self.linear(&x, C, t, &l[0], &mut qkv, 3 * C, Epi::Bias);
                Self::attention(&qkv, &seqs, last, &mut a);
                if !last {
                    self.linear(&a, C, t, &l[1], &mut x, C, Epi::Residual);
                    Self::layernorm(&mut x, C, t, &self.f[&(p.clone() + "ln1.w")], &self.f[&(p.clone() + "ln1.b")], 1e-12, None);
                    self.linear(&x, C, t, &l[2], &mut f, FF, Epi::Gelu);
                    self.linear(&f, FF, t, &l[3], &mut x, C, Epi::Residual);
                    Self::layernorm(&mut x, C, t, &self.f[&(p.clone() + "ln2.w")], &self.f[&(p.clone() + "ln2.b")], 1e-12, None);
                } else {
                    // CLS rows -> compact
                    let mut ac = vec![0f32; b * C];
                    for (j, &(r0, _)) in seqs.iter().enumerate() {
                        ac[j * C..(j + 1) * C].copy_from_slice(&a[r0 * C..(r0 + 1) * C]);
                        xc[j * C..(j + 1) * C].copy_from_slice(&x[r0 * C..(r0 + 1) * C]);
                    }
                    self.linear(&ac, C, b, &l[1], &mut xc, C, Epi::Residual);
                    Self::layernorm(&mut xc, C, b, &self.f[&(p.clone() + "ln1.w")], &self.f[&(p.clone() + "ln1.b")], 1e-12, None);
                    self.linear(&xc, C, b, &l[2], &mut f, FF, Epi::Gelu);
                    self.linear(&f, FF, b, &l[3], &mut xc, C, Epi::Residual);
                    Self::layernorm(&mut xc, C, b, &self.f[&(p.clone() + "ln2.w")], &self.f[&(p.clone() + "ln2.b")], 1e-12, None);
                }
            }
            let mut h1 = vec![0f32; b * P1];
            self.linear(&xc, C, b, &self.t_p1, &mut h1, P1, Epi::Gelu);
            let mut emb = vec![0f32; b * E];
            self.linear(&h1, P1, b, &self.t_p2, &mut emb, E, Epi::Bias);
            Self::normalize(&mut emb);
            emb
        })
    }
}

/// CPUs this process may run on (sched_getaffinity), in order.
fn affinity() -> Vec<usize> {
    std::fs::read_to_string("/proc/self/status").ok().and_then(|s| {
        s.lines().find(|l| l.starts_with("Cpus_allowed_list:")).map(|l| {
            l.split(':').nth(1).unwrap_or("").trim().split(',').flat_map(|r| {
                let mut it = r.split('-').filter_map(|x| x.trim().parse::<usize>().ok());
                match (it.next(), it.next()) {
                    (Some(a), Some(b)) => (a..=b).collect::<Vec<_>>(),
                    (Some(a), None) => vec![a],
                    _ => vec![],
                }
            }).collect()
        })
    }).unwrap_or_default()
}

/// Pin the calling thread to one CPU.
fn pin_to(cpu: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        let _ = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}
