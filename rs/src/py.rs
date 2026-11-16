//! Native Python extension (PyO3, abi3). Tokenization, shape bucketing and text packing all run in Rust; Python
//! passes buffer addresses (host or CUDA memory, e.g. torch's `tensor.data_ptr()`), so no array copies cross the
//! boundary. See python/clipfast_rs_torch.py for a 30-line torch convenience wrapper.
use crate::engine::{Engine as Core, E, IMG};
use crate::frontend::Tok;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

fn err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

/// BiomedCLIP engine. `Engine(assets_dir, fp32=False)`. Not thread-safe: one per thread / GPU.
#[pyclass(unsendable)]
struct Engine {
    eng: Core,
    tok: Tok,
    #[pyo3(get)]
    logit_scale: f64,
    #[pyo3(get)]
    dim: usize,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (assets, fp32 = false))]
    fn new(assets: &str, fp32: bool) -> PyResult<Self> {
        let eng = Core::new(assets, fp32).map_err(err)?;
        let tok = Tok::load(assets).map_err(err)?;
        let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{assets}/config.json"))?)
            .map_err(|e| err(e.into()))?;
        Ok(Engine { eng, tok, logit_scale: cfg["logit_scale"].as_f64().unwrap_or(1.0), dim: E })
    }

    /// Encode `n` preprocessed images. `pixels`: address of n*3*224*224 float32 (host or CUDA, contiguous NCHW);
    /// `out`: address of n*512 float32. Embeddings are L2-normalized.
    fn encode_images_ptr(&self, pixels: usize, n: usize, out: usize) -> PyResult<()> {
        unsafe { self.eng.encode_images_into(pixels as *const f32, n, out as *mut f32, 0) }.map_err(err)
    }

    /// Tokenize (open_clip-identical ids) and encode captions with packed varlen text. `out`: address of
    /// len(texts)*512 float32 (host or CUDA).
    #[pyo3(signature = (texts, out, packed = true))]
    fn encode_texts_ptr(&self, texts: Vec<String>, out: usize, packed: bool) -> PyResult<()> {
        let rows = self.tok.encode(&texts).map_err(err)?;
        self.encode_rows(&rows, out, packed)
    }

    /// Encode pre-tokenized rows ([CLS] ... [SEP], no padding).
    #[pyo3(signature = (rows, out, packed = true))]
    fn encode_token_rows_ptr(&self, rows: Vec<Vec<i32>>, out: usize, packed: bool) -> PyResult<()> {
        self.encode_rows(&rows, out, packed)
    }

    fn tokenize(&self, texts: Vec<String>) -> PyResult<Vec<Vec<i32>>> {
        self.tok.encode(&texts).map_err(err)
    }

    /// (weights MB, workspace MB, whole-device used MB)
    fn memory_mb(&self) -> (f64, f64, f64) {
        self.eng.mem()
    }

    #[getter]
    fn image_floats(&self) -> usize {
        IMG
    }
}

impl Engine {
    fn encode_rows(&self, rows: &[Vec<i32>], out: usize, packed: bool) -> PyResult<()> {
        if rows.is_empty() {
            return Ok(());
        }
        unsafe {
            if packed {
                self.eng.encode_texts_packed_into(rows, out as *mut f32, 0)
            } else {
                self.eng.encode_texts_into(rows, out as *mut f32, 0)
            }
        }
        .map_err(err)
    }
}

/// Load an open_clip checkpoint folder and write runtime assets (no PyTorch needed).
#[pyfunction]
fn export(model_dir: &str, out: &str) -> PyResult<String> {
    crate::export::export(model_dir, out).map(|v| v.to_string()).map_err(err)
}

/// open_clip-identical preprocessing of one encoded image (JPEG/PNG/...): returns 3*224*224 floats.
#[pyfunction]
fn preprocess(data: &[u8]) -> PyResult<Vec<f32>> {
    let mut o = vec![0f32; IMG];
    crate::prep::preprocess_bytes(data, &mut o).map_err(err)?;
    Ok(o)
}

#[pymodule]
fn clipfast_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_function(wrap_pyfunction!(export, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess, m)?)?;
    Ok(())
}
