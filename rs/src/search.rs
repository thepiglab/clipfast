//! Embedding index: int8 coarse search + exact fp32 re-rank (cuda/search.cu).
use anyhow::{bail, Result};
use std::ffi::c_int;

#[repr(C)]
pub struct CS {
    _p: [u8; 0],
}
extern "C" {
    fn cs_create(emb: *const f32, n: i64, d: c_int, rerank: c_int) -> *mut CS;
    fn cs_search(h: *mut CS, q: *const f32, nq: c_int, k: c_int, kc: c_int, exact: c_int, ids: *mut i32, scores: *mut f32) -> c_int;
    fn cs_mem(h: *mut CS, dev: *mut usize, host: *mut usize, ws: *mut usize, sg: *mut f32);
    fn cs_destroy(h: *mut CS);
}

/// Where the fp32 copy used for exact re-ranking lives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rerank {
    /// device memory: fastest
    Device,
    /// pinned host memory read over PCIe (only ~kc rows per query): lowest device memory
    Host,
}

pub struct Index {
    h: *mut CS,
    pub n: usize,
    pub d: usize,
}
unsafe impl Send for Index {}

impl Index {
    /// `emb`: n x d row-major f32 (L2-normalized embeddings).
    pub fn new(emb: &[f32], d: usize, rerank: Rerank) -> Result<Self> {
        let n = emb.len() / d;
        let h = unsafe { cs_create(emb.as_ptr(), n as i64, d as c_int, if rerank == Rerank::Host { 2 } else { 1 }) };
        if h.is_null() {
            bail!("index build failed (d must be a multiple of 16; GPU memory?)");
        }
        Ok(Index { h, n, d })
    }

    /// Top-k (ids, scores) per query, best first. int8 scoring of the whole corpus, then exact fp32 re-rank of the
    /// best `kc` (<= 240) per query. `exact`: fp32 brute force instead (the reference).
    pub fn search(&self, q: &[f32], k: usize, kc: usize, exact: bool) -> Result<(Vec<i32>, Vec<f32>)> {
        let nq = q.len() / self.d;
        let (mut ids, mut sc) = (vec![0i32; nq * k], vec![0f32; nq * k]);
        // large query batches are processed in slices (workspace grows with the slice size)
        let step = 2048usize;
        let mut i = 0;
        while i < nq {
            let m = (nq - i).min(step);
            let rc = unsafe {
                cs_search(self.h, q[i * self.d..].as_ptr(), m as c_int, k as c_int, kc as c_int, exact as c_int,
                          ids[i * k..].as_mut_ptr(), sc[i * k..].as_mut_ptr())
            };
            if rc != 0 {
                bail!("cs_search rc={rc}");
            }
            i += m;
        }
        Ok((ids, sc))
    }

    /// (device MB, pinned host MB, workspace MB, corpus int8 scale)
    pub fn mem(&self) -> (f64, f64, f64, f32) {
        let (mut d, mut h, mut w, mut s) = (0usize, 0usize, 0usize, 0f32);
        unsafe { cs_mem(self.h, &mut d, &mut h, &mut w, &mut s) };
        let mb = |x: usize| x as f64 / (1 << 20) as f64;
        (mb(d), mb(h), mb(w), s)
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        unsafe { cs_destroy(self.h) }
    }
}
