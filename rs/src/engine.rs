//! The GPU engine (CUDA side in cuda/*.cu) and the batching around it: shape buckets, chunking, packed text.
use anyhow::{bail, Context, Result};
use safetensors::SafeTensors;
use std::ffi::{c_char, c_int, c_void};
use std::time::Instant;

#[repr(C)]
pub struct CF {
    _p: [u8; 0],
}
extern "C" {
    // out_rows: rows of embeddings copied to `out` (<= B; the engine may run a padded bucket of B rows)
    fn cf_create(bf16: c_int, max_img: c_int, max_rows: c_int) -> *mut CF;
    fn cf_set_weight(cf: *mut CF, name: *const c_char, host: *const c_void, bytes: usize) -> c_int;
    fn cf_finalize(cf: *mut CF) -> c_int;
    fn cf_set_option(cf: *mut CF, key: *const c_char, value: c_int) -> c_int;
    fn cf_encode_image(cf: *mut CF, px: *const f32, n_in: c_int, b: c_int, out: *mut f32, n_out: c_int, flags: c_int) -> c_int;
    fn cf_encode_text(cf: *mut CF, ids: *const i32, lens: *const i32, b: c_int, l: c_int, out: *mut f32, n_out: c_int, flags: c_int) -> c_int;
    fn cf_encode_text_packed(cf: *mut CF, ids: *const i32, pos: *const i32, cu: *const i32, b: c_int, t: c_int, out: *mut f32, n_out: c_int, flags: c_int) -> c_int;
    fn cf_mem(cf: *mut CF, w: *mut usize, a: *mut usize, used: *mut usize, total: *mut usize) -> c_int;
    fn cf_device_name(buf: *mut c_char, n: c_int) -> c_int;
    fn cf_destroy(cf: *mut CF);
}

pub const E: usize = 512;
pub const IMG: usize = 3 * 224 * 224;
pub const MAX_IMG: usize = 256; // image chunk per graph
pub const MAX_ROWS: usize = 65536; // text tokens per chunk
pub const PAD: i32 = 0;

pub struct Engine {
    cf: *mut CF,
    pub load_s: f64,
}

// The engine owns its CUDA stream/buffers; calls are serialized by the caller (&mut or a Mutex).
unsafe impl Send for Engine {}

impl Engine {
    pub fn new(assets: &str, fp32: bool) -> Result<Self> {
        let t0 = Instant::now();
        let cf = unsafe { cf_create(if fp32 { 0 } else { 1 }, MAX_IMG as c_int, MAX_ROWS as c_int) };
        if cf.is_null() {
            bail!("cf_create failed (out of GPU memory?)");
        }
        let path = format!("{assets}/weights_{}.safetensors", if fp32 { "f32" } else { "bf16" });
        let f = std::fs::File::open(&path).with_context(|| path.clone())?;
        let mm = unsafe { memmap2::Mmap::map(&f)? };
        let st = SafeTensors::deserialize(&mm)?;
        for (name, view) in st.tensors() {
            let cname = std::ffi::CString::new(name.clone())?;
            let d = view.data();
            if unsafe { cf_set_weight(cf, cname.as_ptr(), d.as_ptr() as *const c_void, d.len()) } != 0 {
                bail!("upload {name}");
            }
        }
        if unsafe { cf_finalize(cf) } != 0 {
            bail!("missing weights");
        }
        if std::env::var("CLIPFAST_CUTLASS_GELU").map(|v| v == "0").unwrap_or(false) {
            let k = std::ffi::CString::new("cutlass_gelu")?;
            unsafe { cf_set_option(cf, k.as_ptr(), 0) };
        }
        if std::env::var("CLIPFAST_AUTOTUNE").map(|v| v == "1").unwrap_or(false) {
            let k = std::ffi::CString::new("autotune")?;
            unsafe { cf_set_option(cf, k.as_ptr(), 1) };
        }
        if std::env::var("CLIPFAST_GELU_EPILOGUE").map(|v| v == "1").unwrap_or(false) {
            let k = std::ffi::CString::new("gelu_epilogue")?;
            unsafe { cf_set_option(cf, k.as_ptr(), 1) };
        }
        Ok(Engine { cf, load_s: t0.elapsed().as_secs_f64() })
    }

    /// pixels: n * 3*224*224 f32 (open_clip-preprocessed), host or device pointer. Writes n*512 L2-normalized
    /// embeddings to `out` (host or device). Chunks of <= MAX_IMG run as power-of-two buckets inside the engine.
    /// flags: 1 = reuse the pixels already in the engine buffer (benchmarks), 2 = no CUDA graph.
    ///
    /// # Safety
    /// `px` must point to n*IMG readable f32 and `out` to n*E writable f32 (host or device memory).
    pub unsafe fn encode_images_into(&self, px: *const f32, n: usize, out: *mut f32, flags: i32) -> Result<()> {
        let mut i = 0;
        while i < n {
            let c = (n - i).min(MAX_IMG);
            let nb = if flags & 1 != 0 { c } else { bucket(c) };
            let rc = cf_encode_image(self.cf, px.add(i * IMG), c as c_int, nb as c_int, out.add(i * E), c as c_int,
                                     flags & !4);
            if rc != 0 {
                bail!("cf_encode_image rc={rc}");
            }
            i += c;
        }
        Ok(())
    }

    /// Host convenience wrapper: pixels slice -> Vec of embeddings. flags & 4: skip the copy-out (benchmarks).
    pub fn encode_images(&self, px: &[f32], flags: i32) -> Result<Vec<f32>> {
        let n = px.len() / IMG;
        let mut out = vec![0f32; n * E];
        if flags & 4 != 0 {
            let mut i = 0;
            while i < n {
                let c = (n - i).min(MAX_IMG);
                let nb = if flags & 1 != 0 { c } else { bucket(c) };
                let rc = unsafe { cf_encode_image(self.cf, px[i * IMG..].as_ptr(), c as c_int, nb as c_int,
                                                  std::ptr::null_mut(), 0, flags) };
                if rc != 0 {
                    bail!("cf_encode_image rc={rc}");
                }
                i += c;
            }
            return Ok(out);
        }
        unsafe { self.encode_images_into(px.as_ptr(), n, out.as_mut_ptr(), flags)? };
        Ok(out)
    }

    /// Padded text path. rows: token id sequences (CLS ... SEP, no padding); each chunk is padded to a
    /// (batch bucket, length bucket) with a key-padding mask.
    ///
    /// # Safety
    /// `out` must point to rows.len()*E writable f32 (host or device).
    pub unsafe fn encode_texts_into(&self, rows: &[Vec<i32>], out: *mut f32, flags: i32) -> Result<()> {
        let mut i = 0;
        while i < rows.len() {
            let lmax = rows[i..].iter().take(1024).map(|r| r.len().min(256)).max().unwrap();
            let l = len_bucket(lmax);
            let n = (rows.len() - i).min((MAX_ROWS / l).min(1024)).max(1);
            let lmax = rows[i..i + n].iter().map(|r| r.len().min(256)).max().unwrap();
            let l = len_bucket(lmax);
            let nb = bucket(n).min(MAX_ROWS / l);
            let mut ids = vec![PAD; nb * l];
            let mut lens = vec![1i32; nb];
            for (j, r) in rows[i..i + n].iter().enumerate() {
                let r = &r[..r.len().min(256)];
                ids[j * l..j * l + r.len()].copy_from_slice(r);
                lens[j] = r.len() as i32;
            }
            for j in n..nb {
                ids[j * l] = rows[i][0];
            }
            let (o, no) = if flags & 4 != 0 { (std::ptr::null_mut(), 0) } else { (out.add(i * E), n as c_int) };
            let rc = cf_encode_text(self.cf, ids.as_ptr(), lens.as_ptr(), nb as c_int, l as c_int, o, no, flags);
            if rc != 0 {
                bail!("cf_encode_text rc={rc}");
            }
            i += n;
        }
        Ok(())
    }

    pub fn encode_texts(&self, rows: &[Vec<i32>], flags: i32) -> Result<Vec<f32>> {
        let mut out = vec![0f32; rows.len() * E];
        unsafe { self.encode_texts_into(rows, out.as_mut_ptr(), flags)? };
        Ok(out)
    }

    /// Packed varlen text: only real tokens are computed. Chunks of <= MAX_ROWS tokens and <= 1024 sequences;
    /// sequences longer than 256 tokens are truncated (BiomedCLIP's context length).
    ///
    /// # Safety
    /// `out` must point to rows.len()*E writable f32 (host or device).
    pub unsafe fn encode_texts_packed_into(&self, rows: &[Vec<i32>], out: *mut f32, flags: i32) -> Result<()> {
        let mut i = 0;
        while i < rows.len() {
            let (mut n, mut t) = (0usize, 0usize);
            while i + n < rows.len() && n < 1024 && t + rows[i + n].len().min(256) + 1024 <= MAX_ROWS {
                t += rows[i + n].len().min(256);
                n += 1;
            }
            let nb = bucket(n);
            let tb = tok_bucket(t + (nb - n));
            let (mut ids, mut pos, mut cu) = (vec![0i32; tb], vec![0i32; tb], vec![0i32; nb + 1]);
            let mut off = 0usize;
            for j in 0..nb {
                let r: &[i32] = if j < n { &rows[i + j][..rows[i + j].len().min(256)] } else { &rows[i][..1] };
                ids[off..off + r.len()].copy_from_slice(r);
                for (k, p) in pos[off..off + r.len()].iter_mut().enumerate() {
                    *p = k as i32;
                }
                off += r.len();
                cu[j + 1] = off as i32;
            }
            let (o, no) = if flags & 4 != 0 { (std::ptr::null_mut(), 0) } else { (out.add(i * E), n as c_int) };
            let rc = cf_encode_text_packed(self.cf, ids.as_ptr(), pos.as_ptr(), cu.as_ptr(), nb as c_int, tb as c_int, o,
                                           no, flags);
            if rc != 0 {
                bail!("cf_encode_text_packed rc={rc}");
            }
            i += n;
        }
        Ok(())
    }

    pub fn encode_texts_packed(&self, rows: &[Vec<i32>], flags: i32) -> Result<Vec<f32>> {
        let mut out = vec![0f32; rows.len() * E];
        unsafe { self.encode_texts_packed_into(rows, out.as_mut_ptr(), flags)? };
        Ok(out)
    }

    pub fn mem(&self) -> (f64, f64, f64) {
        let (mut w, mut a, mut u, mut t) = (0usize, 0usize, 0usize, 0usize);
        unsafe { cf_mem(self.cf, &mut w, &mut a, &mut u, &mut t) };
        let mb = |x: usize| x as f64 / (1 << 20) as f64;
        (mb(w), mb(a), mb(u))
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { cf_destroy(self.cf) }
    }
}

pub fn impl_name(fp32: bool) -> String {
    let g = std::env::var("CLIPFAST_GELU_EPILOGUE").map(|v| v == "1").unwrap_or(false);
    let p = std::env::args().any(|a| a == "--packed");
    let nc = std::env::var("CLIPFAST_CUTLASS_GELU").map(|v| v == "0").unwrap_or(false);
    let na = false;
    format!("rs_{}{}{}{}{}", if fp32 { "fp32" } else { "bf16" }, if g { "_geluepi" } else { "" }, if p { "_pack" } else { "" },
            if nc { "_nocutlass" } else { "" }, if na { "_noautotune" } else { "" })
}

pub fn bucket(n: usize) -> usize {
    n.next_power_of_two()
}
/// token-count buckets: round up to a multiple of 2^(floor(log2 T) - 3) (<= 12.5% filler), at least 64
pub fn tok_bucket(t: usize) -> usize {
    let t = t.max(64);
    let step = 1usize << ((usize::BITS - 1 - t.leading_zeros()).saturating_sub(3));
    t.div_ceil(step) * step
}

pub fn len_bucket(l: usize) -> usize {
    for b in [8, 16, 24, 32, 48, 64, 96, 128, 192, 256] {
        if l <= b {
            return b;
        }
    }
    l.div_ceil(64) * 64
}

pub fn device() -> String {
    let mut buf = vec![0u8; 256];
    unsafe { cf_device_name(buf.as_mut_ptr() as *mut c_char, 256) };
    String::from_utf8_lossy(&buf).trim_end_matches('\0').to_string()
}