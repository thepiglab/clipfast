//! Front end: tokenizer (model's own tokenizer.json), image file lists, parallel open_clip-identical preprocessing,
//! .npy output.
use crate::engine::{IMG, PAD};
use crate::prep;
use anyhow::{anyhow, Result};

// ---------------- front end: tokenizer, image files, .npy ----------------
pub struct Tok(tokenizers::Tokenizer);
impl Tok {
    pub fn load(assets: &str) -> Result<Self> {
        let p = format!("{assets}/tokenizer.json");
        let mut t = tokenizers::Tokenizer::from_file(&p).map_err(|e| anyhow!("{p}: {e}"))?;
        t.with_padding(None);
        t.with_truncation(Some(tokenizers::TruncationParams { max_length: 256, ..Default::default() }))
            .map_err(|e| anyhow!("{e}"))?;
        Ok(Tok(t))
    }
    /// open_clip HFTokenizer(clean="whitespace"): collapse whitespace, strip; [CLS] ... [SEP], <= 256 tokens.
    pub fn encode(&self, texts: &[String]) -> Result<Vec<Vec<i32>>> {
        let clean: Vec<String> = texts.iter().map(|t| t.split_whitespace().collect::<Vec<_>>().join(" ")).collect();
        let enc = self.0.encode_batch(clean, true).map_err(|e| anyhow!("{e}"))?;
        Ok(enc.iter().map(|e| e.get_ids().iter().map(|&i| i as i32).filter(|&i| i != PAD).collect()).collect())
    }
}

pub fn write_npy(path: &str, data: &[f32], shape: &[usize]) -> Result<()> {
    use std::io::Write;
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let tuple = if dims.len() == 1 { format!("({},)", dims[0]) } else { format!("({})", dims.join(", ")) };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {tuple}, }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

pub fn image_list(spec: &str) -> Result<Vec<String>> {
    let p = std::path::Path::new(spec);
    if p.is_dir() {
        let exts = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp", "gif"];
        let mut v: Vec<String> = std::fs::read_dir(p)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|q| q.extension().and_then(|e| e.to_str()).is_some_and(|e| exts.contains(&e.to_lowercase().as_str())))
            .map(|q| q.to_string_lossy().to_string())
            .collect();
        v.sort();
        Ok(v)
    } else {
        Ok(std::fs::read_to_string(p)?.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect())
    }
}

pub fn pool(threads: Option<usize>) -> Result<rayon::ThreadPool> {
    let n = threads.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8));
    Ok(rayon::ThreadPoolBuilder::new().num_threads(n).build()?)
}

/// Preprocess a batch of files in parallel. Returns (pixels for successes, ok paths, failures).
pub fn prep_batch(tp: &rayon::ThreadPool, paths: &[String]) -> (Vec<f32>, Vec<String>, Vec<(String, String)>) {
    use rayon::prelude::*;
    let res: Vec<std::result::Result<Vec<f32>, String>> = tp.install(|| {
        paths.par_iter().map(|p| {
            let data = std::fs::read(p).map_err(|e| e.to_string())?;
            let mut out = vec![0f32; IMG];
            prep::preprocess_bytes(&data, &mut out).map_err(|e| e.to_string())?;
            Ok(out)
        }).collect()
    });
    let (mut px, mut ok, mut bad) = (Vec::with_capacity(paths.len() * IMG), vec![], vec![]);
    for (p, r) in paths.iter().zip(res) {
        match r {
            Ok(v) => {
                px.extend_from_slice(&v);
                ok.push(p.clone());
            }
            Err(e) => bad.push((p.clone(), e)),
        }
    }
    (px, ok, bad)
}