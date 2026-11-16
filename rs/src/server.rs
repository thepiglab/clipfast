//! Minimal HTTP embedding server (blocking, one engine behind a mutex, image decode on a CPU thread pool).
//!
//!   GET  /health                      -> {"ok": true, ...}
//!   POST /v1/embed/text   {"texts": ["...", ...]}                       -> {"embeddings": [[512 floats], ...]}
//!   POST /v1/embed/image  raw image bytes (Content-Type image/*)         -> {"embeddings": [[512 floats]]}
//!   POST /v1/embed/image  {"images_b64": ["<base64 jpeg/png>", ...]}    -> {"embeddings": [[...], ...]}
//! Embeddings are L2-normalized; cosine similarity is a dot product; logits = logit_scale * dot.
use crate::engine::{Engine, E, IMG};
use crate::frontend::{pool, Tok};
use crate::prep;
use anyhow::{anyhow, Result};
use base64::Engine as _;
use std::io::Read;
use std::sync::Mutex;

fn reply(req: tiny_http::Request, code: u16, body: serde_json::Value) {
    let h = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
    let _ = req.respond(tiny_http::Response::from_string(body.to_string()).with_status_code(code).with_header(h));
}

fn rows(v: &[f32]) -> Vec<&[f32]> {
    v.chunks(E).collect()
}

pub fn serve(assets: &str, port: u16, threads: Option<usize>, fp32: bool) -> Result<()> {
    let eng = Mutex::new(Engine::new(assets, fp32)?);
    let tok = Tok::load(assets)?;
    let tp = pool(threads)?;
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{assets}/config.json"))?)?;
    let scale = cfg["logit_scale"].as_f64().unwrap_or(1.0);
    let srv = tiny_http::Server::http(("0.0.0.0", port)).map_err(|e| anyhow!("bind :{port}: {e}"))?;
    eprintln!("clipfast-rs serving on :{port} ({} mode, {} decode threads)", if fp32 { "fp32" } else { "bf16" },
              tp.current_num_threads());
    for mut req in srv.incoming_requests() {
        let url = req.url().to_string();
        let mut body = vec![];
        if let Err(e) = req.as_reader().read_to_end(&mut body) {
            reply(req, 400, serde_json::json!({"error": e.to_string()}));
            continue;
        }
        let ctype = req.headers().iter().find(|h| h.field.equiv("Content-Type")).map(|h| h.value.to_string())
            .unwrap_or_default();
        let res: Result<serde_json::Value> = (|| match (req.method(), url.as_str()) {
            (tiny_http::Method::Get, "/health") => {
                Ok(serde_json::json!({"ok": true, "dim": E, "logit_scale": scale, "mode": if fp32 {"fp32"} else {"bf16"}}))
            }
            (tiny_http::Method::Post, "/v1/embed/text") => {
                let v: serde_json::Value = serde_json::from_slice(&body)?;
                let texts: Vec<String> = v["texts"].as_array().ok_or_else(|| anyhow!("expected {{\"texts\": [...]}}"))?
                    .iter().map(|t| t.as_str().unwrap_or("").to_string()).collect();
                let ids = tok.encode(&texts)?;
                let emb = eng.lock().unwrap().encode_texts_packed(&ids, 0)?;
                Ok(serde_json::json!({"embeddings": rows(&emb)}))
            }
            (tiny_http::Method::Post, "/v1/embed/image") => {
                let blobs: Vec<Vec<u8>> = if ctype.starts_with("application/json") {
                    let v: serde_json::Value = serde_json::from_slice(&body)?;
                    v["images_b64"].as_array().ok_or_else(|| anyhow!("expected {{\"images_b64\": [...]}}"))?.iter()
                        .map(|s| base64::engine::general_purpose::STANDARD.decode(s.as_str().unwrap_or("")).map_err(|e| anyhow!(e)))
                        .collect::<Result<_>>()?
                } else {
                    vec![body.clone()]
                };
                use rayon::prelude::*;
                let px: Vec<Vec<f32>> = tp.install(|| blobs.par_iter().map(|b| {
                    let mut o = vec![0f32; IMG];
                    prep::preprocess_bytes(b, &mut o).map(|_| o)
                }).collect::<Result<_>>())?;
                let flat: Vec<f32> = px.concat();
                let emb = eng.lock().unwrap().encode_images(&flat, 0)?;
                Ok(serde_json::json!({"embeddings": rows(&emb)}))
            }
            _ => Err(anyhow!("not found: {url}")),
        })();
        match res {
            Ok(v) => reply(req, 200, v),
            Err(e) => reply(req, if e.to_string().starts_with("not found") { 404 } else { 400 }, serde_json::json!({"error": e.to_string()})),
        }
    }
    Ok(())
}
