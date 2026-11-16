//! clipfast-rs: BiomedCLIP inference in Rust + CUDA (cuBLASLt + custom kernels), bf16 fast or fp32 parity mode.
//!
//!   clipfast-rs parity --assets DIR [--fp32]          compare with open_clip fp32 golden embeddings
//!   clipfast-rs bench  --assets DIR [--fp32] [--img-b 1,16,...] [--txt-b ...] [--txt-l 8,32,...,pmc,mix256]
//! Output: one JSON object per line.
use anyhow::{anyhow, bail, Context, Result};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void};
use std::time::Instant;

#[repr(C)]
struct CF {
    _p: [u8; 0],
}
extern "C" {
    fn cf_create(bf16: c_int, max_img: c_int, max_rows: c_int) -> *mut CF;
    fn cf_set_weight(cf: *mut CF, name: *const c_char, host: *const c_void, bytes: usize) -> c_int;
    fn cf_finalize(cf: *mut CF) -> c_int;
    fn cf_set_option(cf: *mut CF, key: *const c_char, value: c_int) -> c_int;
    fn cf_encode_image(cf: *mut CF, px: *const f32, b: c_int, out: *mut f32, flags: c_int) -> c_int;
    fn cf_encode_text(cf: *mut CF, ids: *const i32, lens: *const i32, b: c_int, l: c_int, out: *mut f32, flags: c_int) -> c_int;
    fn cf_encode_text_packed(cf: *mut CF, ids: *const i32, pos: *const i32, cu: *const i32, b: c_int, t: c_int, out: *mut f32, flags: c_int) -> c_int;
    fn cf_mem(cf: *mut CF, w: *mut usize, a: *mut usize, used: *mut usize, total: *mut usize) -> c_int;
    fn cf_device_name(buf: *mut c_char, n: c_int) -> c_int;
    fn cf_destroy(cf: *mut CF);
}

const E: usize = 512;
const IMG: usize = 3 * 224 * 224;
const MAX_IMG: usize = 256; // image chunk per graph
const MAX_ROWS: usize = 65536; // text tokens per chunk
const PAD: i32 = 0;

struct Engine {
    cf: *mut CF,
    load_s: f64,
}

impl Engine {
    fn new(assets: &str, fp32: bool) -> Result<Self> {
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

    /// pixels: B * 3*224*224 f32 (open_clip-preprocessed). Returns B*512 normalized embeddings.
    fn encode_images(&self, px: &[f32], flags: i32) -> Result<Vec<f32>> {
        let b = px.len() / IMG;
        let mut out = vec![0f32; b * E];
        let mut i = 0;
        while i < b {
            let n = (b - i).min(MAX_IMG);
            let nb = if flags & 1 != 0 { n } else { bucket(n) };
            let rc = if nb == n {
                unsafe { cf_encode_image(self.cf, px[i * IMG..].as_ptr(), n as c_int, out[i * E..].as_mut_ptr(), flags) }
            } else {
                let mut buf = vec![0f32; nb * IMG];
                buf[..n * IMG].copy_from_slice(&px[i * IMG..(i + n) * IMG]);
                let mut o = vec![0f32; nb * E];
                let rc = unsafe { cf_encode_image(self.cf, buf.as_ptr(), nb as c_int, o.as_mut_ptr(), flags) };
                out[i * E..(i + n) * E].copy_from_slice(&o[..n * E]);
                rc
            };
            if rc != 0 {
                bail!("cf_encode_image rc={rc}");
            }
            i += n;
        }
        Ok(out)
    }

    /// rows: token id sequences (CLS ... SEP, no padding). Pads each chunk to (batch bucket, length bucket).
    fn encode_texts(&self, rows: &[Vec<i32>], flags: i32) -> Result<Vec<f32>> {
        let mut out = vec![0f32; rows.len() * E];
        let mut i = 0;
        while i < rows.len() {
            let lmax = rows[i..].iter().take(1024).map(|r| r.len()).max().unwrap();
            let l = len_bucket(lmax);
            let n = (rows.len() - i).min((MAX_ROWS / l).min(1024)).max(1);
            let lmax = rows[i..i + n].iter().map(|r| r.len()).max().unwrap();
            let l = len_bucket(lmax);
            let nb = bucket(n).min(MAX_ROWS / l);
            let mut ids = vec![PAD; nb * l];
            let mut lens = vec![1i32; nb];
            for (j, r) in rows[i..i + n].iter().enumerate() {
                ids[j * l..j * l + r.len()].copy_from_slice(r);
                lens[j] = r.len() as i32;
            }
            for j in n..nb {
                ids[j * l] = rows[i][0];
            }
            let mut o = vec![0f32; nb * E];
            let rc = unsafe { cf_encode_text(self.cf, ids.as_ptr(), lens.as_ptr(), nb as c_int, l as c_int, o.as_mut_ptr(), flags) };
            if rc != 0 {
                bail!("cf_encode_text rc={rc}");
            }
            out[i * E..(i + n) * E].copy_from_slice(&o[..n * E]);
            i += n;
        }
        Ok(out)
    }

    /// Packed varlen: only real tokens are computed. Chunks of <= MAX_ROWS tokens and <= 1024 sequences.
    fn encode_texts_packed(&self, rows: &[Vec<i32>], flags: i32) -> Result<Vec<f32>> {
        let mut out = vec![0f32; rows.len() * E];
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
            let mut o = vec![0f32; nb * E];
            let rc = unsafe {
                cf_encode_text_packed(self.cf, ids.as_ptr(), pos.as_ptr(), cu.as_ptr(), nb as c_int, tb as c_int, o.as_mut_ptr(), flags)
            };
            if rc != 0 {
                bail!("cf_encode_text_packed rc={rc}");
            }
            out[i * E..(i + n) * E].copy_from_slice(&o[..n * E]);
            i += n;
        }
        Ok(out)
    }

    fn mem(&self) -> (f64, f64, f64) {
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

fn impl_name(fp32: bool) -> String {
    let g = std::env::var("CLIPFAST_GELU_EPILOGUE").map(|v| v == "1").unwrap_or(false);
    let p = std::env::args().any(|a| a == "--packed");
    let nc = std::env::var("CLIPFAST_CUTLASS_GELU").map(|v| v == "0").unwrap_or(false);
    let na = false;
    format!("rs_{}{}{}{}{}", if fp32 { "fp32" } else { "bf16" }, if g { "_geluepi" } else { "" }, if p { "_pack" } else { "" },
            if nc { "_nocutlass" } else { "" }, if na { "_noautotune" } else { "" })
}

fn bucket(n: usize) -> usize {
    n.next_power_of_two()
}
/// token-count buckets: round up to a multiple of 2^(floor(log2 T) - 3) (<= 12.5% filler), at least 64
fn tok_bucket(t: usize) -> usize {
    let t = t.max(64);
    let step = 1usize << ((usize::BITS - 1 - t.leading_zeros()).saturating_sub(3));
    t.div_ceil(step) * step
}

fn len_bucket(l: usize) -> usize {
    for b in [8, 16, 24, 32, 48, 64, 96, 128, 192, 256] {
        if l <= b {
            return b;
        }
    }
    l.div_ceil(64) * 64
}

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
}

fn tensor_f32(st: &SafeTensors, n: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let v = st.tensor(n)?;
    Ok((v.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(), v.shape().to_vec()))
}
fn tensor_i32(st: &SafeTensors, n: &str) -> Result<(Vec<i32>, Vec<usize>)> {
    let v = st.tensor(n)?;
    Ok((v.data().chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(), v.shape().to_vec()))
}

fn device() -> String {
    let mut buf = vec![0u8; 256];
    unsafe { cf_device_name(buf.as_mut_ptr() as *mut c_char, 256) };
    String::from_utf8_lossy(&buf).trim_end_matches('\0').to_string()
}

// ---------------- parity ----------------
fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}
fn ranks(row: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&i, &j| row[i].partial_cmp(&row[j]).unwrap());
    let mut r = vec![0f64; row.len()];
    for (k, &i) in idx.iter().enumerate() {
        r[i] = k as f64;
    }
    r
}
fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = ra.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for i in 0..ra.len() {
        sab += (ra[i] - ma) * (rb[i] - mb);
        saa += (ra[i] - ma).powi(2);
        sbb += (rb[i] - mb).powi(2);
    }
    sab / (saa.sqrt() * sbb.sqrt())
}
fn argmax(v: &[f64]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

fn parity(args: &[String]) -> Result<()> {
    let assets = arg(args, "--assets").ok_or(anyhow!("--assets"))?;
    let fp32 = args.iter().any(|a| a == "--fp32");
    let eng = Engine::new(&assets, fp32)?;
    let g = std::fs::read(format!("{assets}/golden.safetensors"))?;
    let st = SafeTensors::deserialize(&g)?;
    let (px, ps) = tensor_f32(&st, "pixels")?;
    let (ids, is) = tensor_i32(&st, "ids")?;
    let (ri, _) = tensor_f32(&st, "img_emb")?;
    let (rt, _) = tensor_f32(&st, "txt_emb")?;
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{assets}/config.json"))?)?;
    let scale = cfg["logit_scale"].as_f64().unwrap_or(100.0);
    let (ni, nt, l) = (ps[0], is[0], is[1]);
    let rows: Vec<Vec<i32>> = (0..nt).map(|i| ids[i * l..(i + 1) * l].iter().cloned().take_while(|&t| t != PAD).collect()).collect();
    // whole batch + singles (first 4 images, first 3 texts), worst case per row
    let eb = eng.encode_images(&px, 0)?;
    let packed = args.iter().any(|a| a == "--packed");
    let enc_t = |r: &[Vec<i32>]| if packed { eng.encode_texts_packed(r, 0) } else { eng.encode_texts(r, 0) };
    let tb = enc_t(&rows)?;
    let mut ei = eb.clone();
    for i in 0..4 {
        let o = eng.encode_images(&px[i * IMG..(i + 1) * IMG], 0)?;
        if dot(&o, &ri[i * E..(i + 1) * E]) < dot(&ei[i * E..(i + 1) * E], &ri[i * E..(i + 1) * E]) {
            ei[i * E..(i + 1) * E].copy_from_slice(&o);
        }
    }
    let mut et = tb.clone();
    for i in 0..3 {
        let o = enc_t(&rows[i..i + 1])?;
        if dot(&o, &rt[i * E..(i + 1) * E]) < dot(&et[i * E..(i + 1) * E], &rt[i * E..(i + 1) * E]) {
            et[i * E..(i + 1) * E].copy_from_slice(&o);
        }
    }
    let cos = |a: &[f32], b: &[f32], n: usize| -> (f64, f64, f64) {
        let mut mn = f64::MAX;
        let mut sum = 0.0;
        let mut mx_abs = 0f64;
        for i in 0..n {
            let c = dot(&a[i * E..(i + 1) * E], &b[i * E..(i + 1) * E]);
            mn = mn.min(c);
            sum += c;
            for k in 0..E {
                mx_abs = mx_abs.max((a[i * E + k] - b[i * E + k]).abs() as f64);
            }
        }
        (mn, sum / n as f64, mx_abs)
    };
    let (icmin, icmean, imax) = cos(&ei, &ri, ni);
    let (tcmin, tcmean, tmax) = cos(&et, &rt, nt);
    let sim = |a: &[f32], b: &[f32]| -> Vec<Vec<f64>> {
        (0..ni).map(|i| (0..nt).map(|j| dot(&a[i * E..(i + 1) * E], &b[j * E..(j + 1) * E])).collect()).collect()
    };
    let (es, rs) = (sim(&ei, &et), sim(&ri, &rt));
    let mut logit_max = 0f64;
    let (mut i2t, mut sp_i2t, mut top5) = (0, 0.0, 0usize);
    for i in 0..ni {
        for j in 0..nt {
            logit_max = logit_max.max(scale * (es[i][j] - rs[i][j]).abs());
        }
        i2t += (argmax(&es[i]) == argmax(&rs[i])) as usize;
        sp_i2t += spearman(&es[i], &rs[i]);
        let top = |v: &Vec<f64>| {
            let mut ix: Vec<usize> = (0..v.len()).collect();
            ix.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
            ix.truncate(5);
            ix
        };
        let (a, b) = (top(&es[i]), top(&rs[i]));
        top5 += a.iter().filter(|x| b.contains(x)).count();
    }
    let col = |m: &Vec<Vec<f64>>, j: usize| -> Vec<f64> { (0..ni).map(|i| m[i][j]).collect() };
    let (mut t2i, mut sp_t2i) = (0, 0.0);
    for j in 0..nt {
        let (a, b) = (col(&es, j), col(&rs, j));
        t2i += (argmax(&a) == argmax(&b)) as usize;
        sp_t2i += spearman(&a, &b);
    }
    let zs = (0..9).filter(|&i| argmax(&es[i][..9].to_vec()) == argmax(&rs[i][..9].to_vec())).count();
    let (w, a, u) = eng.mem();
    println!(
        "{}",
        serde_json::json!({"kind": "acc", "impl": impl_name(fp32), "gpu": device(),
            "img_cos_min": icmin, "img_cos_mean": icmean, "txt_cos_min": tcmin, "txt_cos_mean": tcmean,
            "img_maxabs": imax, "txt_maxabs": tmax, "logit_maxabs": logit_max,
            "i2t_top1_agree": format!("{i2t}/{ni}"), "t2i_top1_agree": format!("{t2i}/{nt}"),
            "i2t_top5_overlap": top5 as f64 / (5 * ni) as f64,
            "i2t_rank_spearman": sp_i2t / ni as f64, "t2i_rank_spearman": sp_t2i / nt as f64,
            "zeroshot_agree": format!("{zs}/9"), "weights_mb": w, "workspace_mb": a, "device_used_mb": u,
            "load_s": eng.load_s})
    );
    Ok(())
}

// ---------------- bench ----------------
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unif(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.unif().max(1e-12), self.unif());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn time_ms(mut f: impl FnMut() -> Result<()>) -> Result<(f64, f64, f64)> {
    for _ in 0..3 {
        f()?;
    }
    let t0 = Instant::now();
    f()?;
    let one = t0.elapsed().as_secs_f64() * 1e3;
    let n = ((300.0 / one.max(1e-3)) as usize).max(1);
    let mut v = vec![];
    for _ in 0..5 {
        let t0 = Instant::now();
        for _ in 0..n {
            f()?;
        }
        v.push(t0.elapsed().as_secs_f64() * 1e3 / n as f64);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((v[2], v[0], v[4]))
}

fn list(args: &[String], k: &str, d: &str) -> Vec<String> {
    arg(args, k).unwrap_or(d.into()).split(',').map(|s| s.to_string()).collect()
}

fn bench(args: &[String]) -> Result<()> {
    let assets = arg(args, "--assets").ok_or(anyhow!("--assets"))?;
    let fp32 = args.iter().any(|a| a == "--fp32");
    let name = impl_name(fp32);
    let eng = Engine::new(&assets, fp32)?;
    let dev = device();
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let (w, a, u) = eng.mem();
    println!("{}", serde_json::json!({"kind": "build", "impl": name, "gpu": dev, "weights_mb": w, "workspace_mb": a,
        "device_used_mb": u, "load_s": eng.load_s}));
    for bs in list(args, "--img-b", "1,4,16,64,256,1024") {
        let b: usize = bs.parse()?;
        let px: Vec<f32> = (0..b * IMG).map(|_| rng.normal() as f32).collect();
        let t0 = Instant::now();
        eng.encode_images(&px, 0)?;
        let first = t0.elapsed().as_secs_f64() * 1e3;
        let (med, lo, hi) = time_ms(|| eng.encode_images(&px, 5).map(|_| ()))?; // device-resident input, no D2H (GPU forward)
        let (e2e, _, _) = time_ms(|| eng.encode_images(&px, 0).map(|_| ()))?;
        let (_, _, used) = eng.mem();
        println!("{}", serde_json::json!({"kind": "image", "impl": name, "gpu": dev, "B": b, "ms": med, "ms_min": lo,
            "ms_max": hi, "img_per_s": b as f64 / med * 1e3, "e2e_ms": e2e, "first_call_ms": first,
            "weights_mb": w, "workspace_mb": a, "device_used_mb": used}));
    }
    let vocab = 30000i32;
    for bs in list(args, "--txt-b", "1,16,256,1024") {
        let b: usize = bs.parse()?;
        for ls in list(args, "--txt-l", "8,32,128,256,mix256,pmc") {
            let lens: Vec<usize> = (0..b).map(|_| match ls.as_str() {
                "pmc" => (40.0 * (0.8 * rng.normal()).exp()).round().clamp(5.0, 256.0) as usize,
                s if s.starts_with("mix") => {
                    let l: usize = s[3..].parse().unwrap();
                    l / 2 + (rng.next() as usize % (l - l / 2 + 1))
                }
                s => s.parse().unwrap(),
            }).collect();
            let rows: Vec<Vec<i32>> = lens.iter().map(|&n| {
                let mut r = vec![2i32];
                r.extend((0..n.saturating_sub(2)).map(|_| 1000 + (rng.next() % (vocab as u64 - 1000)) as i32));
                r.push(3);
                r
            }).collect();
            let packed = args.iter().any(|a| a == "--packed");
            let enc = |f: i32| if packed { eng.encode_texts_packed(&rows, f) } else { eng.encode_texts(&rows, f) };
            let (med, lo, hi) = time_ms(|| enc(5).map(|_| ()))?;
            let (e2e, _, _) = time_ms(|| enc(0).map(|_| ()))?;
            let (_, _, used) = eng.mem();
            println!("{}", serde_json::json!({"kind": "text", "impl": name, "gpu": dev, "B": b, "L": ls, "ms": med,
                "ms_min": lo, "ms_max": hi, "txt_per_s": b as f64 / med * 1e3, "e2e_ms": e2e,
                "weights_mb": w, "workspace_mb": a, "device_used_mb": used}));
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("parity") => parity(&args),
        Some("bench") => bench(&args),
        _ => {
            eprintln!("usage: clipfast-rs parity|bench --assets DIR [--fp32] ...");
            let _ = HashMap::<u8, u8>::new();
            Ok(())
        }
    }
}
