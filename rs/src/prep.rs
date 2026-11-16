//! Image preprocessing identical to open_clip's eval transform on a PIL image opened with
//! `Image.open(path).convert("RGB")`:
//!   torchvision Resize(224, BICUBIC) on PIL (shortest side -> 224, long side int(224 * long / short)),
//!   CenterCrop(224) (offsets int(round((n - 224) / 2)), Python round-half-even), ToTensor (/255), Normalize.
//! The resize re-implements Pillow's Resample.c 8-bit path: bicubic a = -0.5, support scaled by the downscale factor,
//! coefficients normalized then quantized to 22-bit fixed point, separable horizontal-then-vertical passes with
//! rounding and uint8 clipping between them. JPEG is decoded with libjpeg-turbo (mozjpeg crate, default ISLOW DCT and
//! fancy upsampling, as Pillow), other formats with the `image` crate. No EXIF rotation and alpha dropped, as
//! `convert("RGB")` does.
use anyhow::{anyhow, Result};

pub const SIZE: usize = 224;
pub const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
pub const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
const PRECISION_BITS: i32 = 32 - 8 - 2;

/// Decode to packed RGB8 (w, h, data).
pub fn decode_rgb(data: &[u8]) -> Result<(usize, usize, Vec<u8>)> {
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        let r = std::panic::catch_unwind(|| -> std::io::Result<(usize, usize, Vec<u8>)> {
            let d = mozjpeg::Decompress::new_mem(data)?;
            let mut s = d.rgb()?;
            let (w, h) = (s.width(), s.height());
            let px: Vec<[u8; 3]> = s.read_scanlines()?;
            s.finish()?;
            Ok((w, h, px.into_iter().flatten().collect()))
        });
        if let Ok(Ok(im)) = r {
            return Ok(im);
        } // e.g. CMYK JPEG: fall through to the image crate
    }
    let mut r = image::ImageReader::new(std::io::Cursor::new(data)).with_guessed_format()?;
    r.no_limits();
    let im = r.decode().map_err(|e| anyhow!("image decode: {e}"))?;
    let (w, h) = (im.width() as usize, im.height() as usize);
    Ok((w, h, im.to_rgb8().into_raw())) // alpha dropped, gray replicated: as PIL convert("RGB")
}

fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Pillow precompute_coeffs + normalize_coeffs_8bpc. Returns (ksize, bounds (xmin, n), int coefficients).
fn coeffs(in_size: usize, out_size: usize) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
    let (in0, in1) = (0.0f64, in_size as f64);
    let scale = (in1 - in0) / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut kk = vec![0i32; out_size * ksize];
    let mut k = vec![0f64; ksize];
    for xx in 0..out_size {
        let center = in0 + (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        let mut xmin = (center - support + 0.5) as i64; // C (int) cast truncates toward zero
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i64;
        if xmax > in_size as i64 {
            xmax = in_size as i64;
        }
        let n = (xmax - xmin) as usize;
        let mut ww = 0.0;
        for x in 0..n {
            let w = bicubic((x as f64 + xmin as f64 - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for x in 0..ksize {
            let v = if x < n { if ww != 0.0 { k[x] / ww } else { k[x] } } else { 0.0 };
            let f = v * (1i64 << PRECISION_BITS) as f64;
            kk[xx * ksize + x] = if v < 0.0 { (-0.5 + f) as i32 } else { (0.5 + f) as i32 };
        }
        bounds.push((xmin as usize, n));
    }
    (ksize, bounds, kk)
}

#[inline]
fn clip8(v: i32) -> u8 {
    let s = v >> PRECISION_BITS; // arithmetic shift, as the C lookup index
    s.clamp(0, 255) as u8
}

/// Pillow Image.resize((ow, oh), BICUBIC) for RGB8.
pub fn resize_bicubic(src: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    if ow == w && oh == h {
        return src.to_vec();
    }
    let need_h = ow != w;
    let need_v = oh != h;
    let (ksv, mut bv, kv) = coeffs(h, oh);
    let (ksh, bh, kh) = coeffs(w, ow);
    let ybox_first = bv[0].0;
    let ybox_last = bv[oh - 1].0 + bv[oh - 1].1;
    let (mut cur, mut cw, mut _ch) = (src.to_vec(), w, h);
    let mut row_off = 0usize;
    if need_h {
        for b in bv.iter_mut() {
            b.0 -= ybox_first;
        }
        let rows = ybox_last - ybox_first;
        let mut tmp = vec![0u8; rows * ow * 3];
        for yy in 0..rows {
            let srow = &src[(yy + ybox_first) * w * 3..(yy + ybox_first + 1) * w * 3];
            for xx in 0..ow {
                let (xmin, n) = bh[xx];
                let k = &kh[xx * ksh..xx * ksh + n];
                let (mut s0, mut s1, mut s2) = (1i32 << (PRECISION_BITS - 1), 1i32 << (PRECISION_BITS - 1), 1i32 << (PRECISION_BITS - 1));
                for (x, &kx) in k.iter().enumerate() {
                    let p = (xmin + x) * 3;
                    s0 += srow[p] as i32 * kx;
                    s1 += srow[p + 1] as i32 * kx;
                    s2 += srow[p + 2] as i32 * kx;
                }
                let o = (yy * ow + xx) * 3;
                tmp[o] = clip8(s0);
                tmp[o + 1] = clip8(s1);
                tmp[o + 2] = clip8(s2);
            }
        }
        cur = tmp;
        cw = ow;
        _ch = rows;
    } else {
        row_off = 0;
    }
    if !need_v {
        return cur;
    }
    let mut out = vec![0u8; ow * oh * 3];
    for yy in 0..oh {
        let (ymin, n) = bv[yy];
        let ymin = ymin + row_off;
        let k = &kv[yy * ksv..yy * ksv + n];
        for xx in 0..cw {
            let (mut s0, mut s1, mut s2) = (1i32 << (PRECISION_BITS - 1), 1i32 << (PRECISION_BITS - 1), 1i32 << (PRECISION_BITS - 1));
            for (y, &ky) in k.iter().enumerate() {
                let p = ((ymin + y) * cw + xx) * 3;
                s0 += cur[p] as i32 * ky;
                s1 += cur[p + 1] as i32 * ky;
                s2 += cur[p + 2] as i32 * ky;
            }
            let o = (yy * ow + xx) * 3;
            out[o] = clip8(s0);
            out[o + 1] = clip8(s1);
            out[o + 2] = clip8(s2);
        }
    }
    out
}

/// Python round() for x.0 / x.5 values: round half to even.
fn round_half_even(x: f64) -> i64 {
    let f = x.floor();
    let d = x - f;
    if d > 0.5 || (d == 0.5 && (f as i64) % 2 != 0) {
        f as i64 + 1
    } else {
        f as i64
    }
}

/// Full open_clip eval transform from RGB8 -> CHW float32 normalized [3*224*224], written into `out`.
pub fn preprocess_rgb(w: usize, h: usize, rgb: &[u8], out: &mut [f32]) {
    // torchvision _compute_resized_output_size for size=[224]
    let (short, long) = if w <= h { (w, h) } else { (h, w) };
    let new_long = ((SIZE * long) as f64 / short as f64) as usize;
    let (nw, nh) = if w <= h { (SIZE, new_long) } else { (new_long, SIZE) };
    let r = resize_bicubic(rgb, w, h, nw, nh);
    let top = round_half_even((nh as f64 - SIZE as f64) / 2.0).max(0) as usize;
    let left = round_half_even((nw as f64 - SIZE as f64) / 2.0).max(0) as usize;
    for c in 0..3 {
        for y in 0..SIZE {
            let sy = y + top;
            for x in 0..SIZE {
                let v = if sy < nh && x + left < nw { r[(sy * nw + x + left) * 3 + c] } else { 0 };
                out[(c * SIZE + y) * SIZE + x] = (v as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
}

pub fn preprocess_bytes(data: &[u8], out: &mut [f32]) -> Result<()> {
    let (w, h, rgb) = decode_rgb(data)?;
    if w == 0 || h == 0 {
        return Err(anyhow!("empty image"));
    }
    preprocess_rgb(w, h, &rgb, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rounding() {
        assert_eq!(round_half_even(0.5), 0);
        assert_eq!(round_half_even(1.5), 2);
        assert_eq!(round_half_even(2.0), 2);
    }
    #[test]
    fn identity() {
        let v: Vec<u8> = (0..(4 * 3 * 3)).map(|i| i as u8).collect();
        assert_eq!(resize_bicubic(&v, 4, 3, 4, 3), v);
    }
}
