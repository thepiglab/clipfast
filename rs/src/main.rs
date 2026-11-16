//! clipfast-rs: BiomedCLIP inference in Rust + CUDA (cuBLASLt + custom kernels), bf16 fast or fp32 parity mode.
//!
//!   clipfast-rs parity --assets DIR [--fp32]          compare with open_clip fp32 golden embeddings
//!   clipfast-rs bench  --assets DIR [--fp32] [--img-b 1,16,...] [--txt-b ...] [--txt-l 8,32,...,pmc,mix256]
//!   clipfast-rs embed  --assets DIR [--images DIR|LIST.txt] [--texts FILE] --out PREFIX [--threads N] [--fp32]
//!                      -> PREFIX.images.npy / .images.txt, PREFIX.texts.npy (float32 [N,512], L2-normalized)
//!   clipfast-rs prep   --images DIR|LIST.txt --out FILE.npy     (open_clip-identical preprocessed pixels, for tests)
//!   clipfast-rs tokenize --assets DIR --texts FILE               (token ids as JSON lines, for tests)
//! Output: one JSON object per line.
//!   clipfast-rs export --model-dir DIR --out DIR                 (open_clip .bin -> safetensors + config + tokenizer)
//!   clipfast-rs serve  --assets DIR [--port 8080] [--threads N]  (HTTP: POST /v1/embed/text, /v1/embed/image)
//!   clipfast-rs search --corpus EMB.npy [--names NAMES.txt] (--queries Q.npy | --assets DIR --texts FILE)
//!                      [--k 10] [--kc 200] [--rerank device|host] [--exact]   (int8 search + exact fp32 re-rank)
//!   clipfast-rs search-bench --assets DIR [--real 200000] [--synth 1800000] [--nq 1,8,256,2048] [--k 10]
use anyhow::{anyhow, Result};
use clipfast_rs::engine::*;
use clipfast_rs::frontend::*;
use clipfast_rs::search::{Index, Rerank};
use clipfast_rs::{export, server};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::time::Instant;

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

fn cmd_prep(args: &[String]) -> Result<()> {
    let list = image_list(&arg(args, "--images").ok_or(anyhow!("--images"))?)?;
    let out = arg(args, "--out").ok_or(anyhow!("--out"))?;
    let tp = pool(arg(args, "--threads").map(|s| s.parse()).transpose()?)?;
    let (px, ok, bad) = prep_batch(&tp, &list);
    write_npy(&out, &px, &[ok.len(), 3, 224, 224])?;
    std::fs::write(format!("{out}.txt"), ok.join("\n"))?;
    println!("{}", serde_json::json!({"kind": "prep", "ok": ok.len(), "failed": bad}));
    Ok(())
}

fn cmd_tokenize(args: &[String]) -> Result<()> {
    let tok = Tok::load(&arg(args, "--assets").ok_or(anyhow!("--assets"))?)?;
    let texts: Vec<String> = std::fs::read_to_string(arg(args, "--texts").ok_or(anyhow!("--texts"))?)?
        .lines().map(|s| s.to_string()).collect();
    for ids in tok.encode(&texts)? {
        println!("{}", serde_json::json!(ids));
    }
    Ok(())
}

fn cmd_embed(args: &[String]) -> Result<()> {
    let assets = arg(args, "--assets").ok_or(anyhow!("--assets"))?;
    let out = arg(args, "--out").ok_or(anyhow!("--out"))?;
    let fp32 = args.iter().any(|a| a == "--fp32");
    let batch: usize = arg(args, "--batch").map(|s| s.parse()).transpose()?.unwrap_or(256);
    let eng = Engine::new(&assets, fp32)?;
    let tp = pool(arg(args, "--threads").map(|s| s.parse()).transpose()?)?;
    let mut summary = serde_json::json!({"kind": "embed", "impl": impl_name(fp32), "gpu": device(),
        "threads": tp.current_num_threads(), "load_s": eng.load_s});
    if let Some(spec) = arg(args, "--images") {
        let list = image_list(&spec)?;
        let t0 = Instant::now();
        let (mut emb, mut names, mut failed) = (Vec::with_capacity(list.len() * E), vec![], vec![]);
        let (mut cpu_s, mut gpu_s) = (0f64, 0f64);
        // pipeline: CPU threads preprocess chunk i+1 while the GPU encodes chunk i
        std::thread::scope(|sc| -> Result<()> {
            let (tx, rx) = std::sync::mpsc::sync_channel::<(Vec<f32>, Vec<String>, Vec<(String, String)>, f64)>(2);
            let tp = &tp;
            let list = &list;
            sc.spawn(move || {
                for chunk in list.chunks(batch) {
                    let t = Instant::now();
                    let (px, ok, bad) = prep_batch(tp, chunk);
                    if tx.send((px, ok, bad, t.elapsed().as_secs_f64())).is_err() {
                        break;
                    }
                }
            });
            for (px, ok, bad, ct) in rx {
                cpu_s += ct;
                let t = Instant::now();
                if !ok.is_empty() {
                    emb.extend(eng.encode_images(&px, 0)?);
                }
                gpu_s += t.elapsed().as_secs_f64();
                names.extend(ok);
                failed.extend(bad);
            }
            Ok(())
        })?;
        let wall = t0.elapsed().as_secs_f64();
        write_npy(&format!("{out}.images.npy"), &emb, &[names.len(), E])?;
        std::fs::write(format!("{out}.images.txt"), names.join("\n"))?;
        summary["images"] = serde_json::json!({"n": names.len(), "failed": failed.len(), "wall_s": wall,
            "img_per_s": names.len() as f64 / wall, "cpu_prep_s": cpu_s, "gpu_encode_s": gpu_s,
            "failed_examples": failed.iter().take(5).collect::<Vec<_>>()});
    }
    if let Some(tf) = arg(args, "--texts") {
        let tok = Tok::load(&assets)?;
        let texts: Vec<String> = std::fs::read_to_string(&tf)?.lines().map(|s| s.to_string()).collect();
        let t0 = Instant::now();
        let rows = tok.encode(&texts)?;
        let tok_s = t0.elapsed().as_secs_f64();
        let emb = eng.encode_texts_packed(&rows, 0)?;
        let wall = t0.elapsed().as_secs_f64();
        write_npy(&format!("{out}.texts.npy"), &emb, &[rows.len(), E])?;
        summary["texts"] = serde_json::json!({"n": rows.len(), "wall_s": wall, "tokenize_s": tok_s,
            "txt_per_s": rows.len() as f64 / wall});
    }
    let (w, a, u) = eng.mem();
    summary["weights_mb"] = serde_json::json!(w);
    summary["workspace_mb"] = serde_json::json!(a);
    summary["device_used_mb"] = serde_json::json!(u);
    println!("{summary}");
    Ok(())
}

// ---------------- search ----------------
fn rerank_arg(args: &[String]) -> Rerank {
    if arg(args, "--rerank").as_deref() == Some("host") { Rerank::Host } else { Rerank::Device }
}

fn cmd_search(args: &[String]) -> Result<()> {
    let (corpus, shape) = read_npy(&arg(args, "--corpus").ok_or(anyhow!("--corpus"))?)?;
    let d = *shape.last().unwrap_or(&512);
    let names: Option<Vec<String>> = arg(args, "--names").map(|p| std::fs::read_to_string(p).map(|s| s.lines().map(String::from).collect())).transpose()?;
    let k: usize = arg(args, "--k").map(|s| s.parse()).transpose()?.unwrap_or(10);
    let kc: usize = arg(args, "--kc").map(|s| s.parse()).transpose()?.unwrap_or(200).max(k);
    let (q, labels): (Vec<f32>, Vec<String>) = if let Some(qp) = arg(args, "--queries") {
        let (q, s) = read_npy(&qp)?;
        let n = q.len() / *s.last().unwrap_or(&d);
        (q, (0..n).map(|i| i.to_string()).collect())
    } else {
        let assets = arg(args, "--assets").ok_or(anyhow!("--assets or --queries"))?;
        let texts: Vec<String> = std::fs::read_to_string(arg(args, "--texts").ok_or(anyhow!("--texts"))?)?
            .lines().map(String::from).collect();
        let eng = Engine::new(&assets, false)?;
        let rows = Tok::load(&assets)?.encode(&texts)?;
        (eng.encode_texts_packed(&rows, 0)?, texts)
    };
    let idx = Index::new(&corpus, d, rerank_arg(args))?;
    let t0 = Instant::now();
    let (ids, sc) = idx.search(&q, k, kc, args.iter().any(|a| a == "--exact"))?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    for (qi, lab) in labels.iter().enumerate() {
        let hits: Vec<serde_json::Value> = (0..k).map(|j| {
            let id = ids[qi * k + j];
            serde_json::json!({"id": id, "score": sc[qi * k + j],
                "name": names.as_ref().and_then(|n| n.get(id.max(0) as usize)).cloned()})
        }).collect();
        println!("{}", serde_json::json!({"query": lab, "hits": hits}));
    }
    let (dm, hm, wm, _) = idx.mem();
    eprintln!("{}", serde_json::json!({"kind": "search", "corpus": idx.n, "queries": labels.len(), "ms": ms,
        "device_mb": dm, "host_mb": hm, "workspace_mb": wm}));
    Ok(())
}

fn cmd_search_bench(args: &[String]) -> Result<()> {
    let assets = arg(args, "--assets").ok_or(anyhow!("--assets"))?;
    let n_real: usize = arg(args, "--real").map(|s| s.parse()).transpose()?.unwrap_or(200_000);
    let n_syn: usize = arg(args, "--synth").map(|s| s.parse()).transpose()?.unwrap_or(1_800_000);
    let k: usize = arg(args, "--k").map(|s| s.parse()).transpose()?.unwrap_or(10);
    let kc: usize = arg(args, "--kc").map(|s| s.parse()).transpose()?.unwrap_or(200);
    let rr = rerank_arg(args);
    let eng = Engine::new(&assets, false)?;
    let mut rng = Rng(0x2545F4914F6CDD1D);
    let caption = |rng: &mut Rng| -> Vec<i32> {
        let n = (40.0 * (0.8 * rng.normal()).exp()).round().clamp(5.0, 256.0) as usize;
        let mut r = vec![2i32];
        r.extend((0..n - 2).map(|_| 1000 + (rng.next() % 29000) as i32));
        r.push(3);
        r
    };
    // corpus: real BiomedCLIP text embeddings + clustered synthetic unit vectors
    let rows: Vec<Vec<i32>> = (0..n_real).map(|_| caption(&mut rng)).collect();
    let t0 = Instant::now();
    let mut corpus = eng.encode_texts_packed(&rows, 0)?;
    let enc_s = t0.elapsed().as_secs_f64();
    const D: usize = 512;
    let centers: Vec<f32> = (0..1000 * D).map(|_| rng.normal() as f32).collect();
    for i in 0..n_syn {
        let c = (rng.next() % 1000) as usize;
        let mut v: Vec<f32> = (0..D).map(|t| centers[c * D + t] + 0.6 * rng.normal() as f32).collect();
        let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= nrm);
        corpus.extend_from_slice(&v);
        let _ = i;
    }
    // queries: fresh text embeddings (text->text) and random-pixel image embeddings (image->text)
    let maxq = list(args, "--nq", "1,8,256,2048").iter().map(|s| s.parse::<usize>().unwrap()).max().unwrap();
    let qrows: Vec<Vec<i32>> = (0..maxq).map(|_| caption(&mut rng)).collect();
    let mut queries = eng.encode_texts_packed(&qrows, 0)?;
    let px: Vec<f32> = (0..64 * IMG).map(|_| rng.normal() as f32).collect();
    let qi = eng.encode_images(&px, 0)?;
    // every other query among the first 128 is an image embedding (image -> text retrieval)
    for v in 0..64usize.min(maxq / 2) {
        let j = 2 * v + 1;
        queries[j * D..(j + 1) * D].copy_from_slice(&qi[v * D..(v + 1) * D]);
    }
    drop(eng);
    let t0 = Instant::now();
    let idx = Index::new(&corpus, D, rr)?;
    let build_s = t0.elapsed().as_secs_f64();
    let (dm, hm, wm, sg) = idx.mem();
    let n = idx.n as f64 / 1e6;
    println!("{}", serde_json::json!({"kind": "index", "n": idx.n, "real": n_real, "synthetic": n_syn, "encode_s": enc_s,
        "build_s": build_s, "rerank": format!("{rr:?}"), "device_mb": dm, "host_mb": hm, "device_mb_per_million": dm / n,
        "host_mb_per_million": hm / n, "int8_scale": sg}));
    // CPU brute-force reference for the first 2 queries (validates the GPU exact path itself)
    {
        let q2 = &queries[..2 * D];
        let (ge, _) = idx.search(q2, k, kc, true)?;
        let (gq, _) = idx.search(q2, k, kc, false)?;
        for r in 0..2 {
            let qv = &q2[r * D..(r + 1) * D];
            let mut s: Vec<(f32, i32)> = corpus.chunks_exact(D).enumerate()
                .map(|(i, x)| (x.iter().zip(qv).map(|(a, b)| a * b).sum::<f32>(), i as i32)).collect();
            s.select_nth_unstable_by(k, |a, b| b.0.partial_cmp(&a.0).unwrap());
            s.truncate(k);
            s.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let cpu: Vec<i32> = s.iter().map(|x| x.1).collect();
            println!("{}", serde_json::json!({"kind": "cpu_check", "query": r, "cpu_top": cpu,
                "gpu_exact_matches_cpu": cpu == ge[r * k..(r + 1) * k].to_vec(),
                "gpu_int8_rerank_matches_cpu": cpu == gq[r * k..(r + 1) * k].to_vec(), "cpu_best": s[0].0}));
        }
    }
    for nqs in list(args, "--nq", "1,8,256,2048") {
        let nq: usize = nqs.parse()?;
        let q = &queries[..nq * D];
        let timed = |exact: bool| -> Result<(f64, Vec<i32>, Vec<f32>)> {
            let (ids, sc) = idx.search(q, k, kc, exact)?;   // warm (workspace growth)
            let reps = (2000 / nq).clamp(3, 50);
            let t0 = Instant::now();
            for _ in 0..reps {
                idx.search(q, k, kc, exact)?;
            }
            Ok((t0.elapsed().as_secs_f64() * 1e3 / reps as f64, ids, sc))
        };
        let (ms_exact, ide, sce) = timed(true)?;
        let (ms_q, idq, scq) = timed(false)?;
        let (mut same, mut overlap, mut max_sdiff) = (0usize, 0usize, 0f32);
        for r in 0..nq {
            let a = &ide[r * k..(r + 1) * k];
            let b = &idq[r * k..(r + 1) * k];
            same += (a == b) as usize;
            overlap += a.iter().filter(|x| b.contains(x)).count();
            for j in 0..k {
                max_sdiff = max_sdiff.max((sce[r * k + j] - scq[r * k + j]).abs());
            }
        }
        println!("{}", serde_json::json!({"kind": "search", "nq": nq, "k": k, "kc": kc, "rerank": format!("{rr:?}"),
            "exact_fp32_ms": ms_exact, "int8_rerank_ms": ms_q, "speedup": ms_exact / ms_q,
            "topk_identical": format!("{same}/{nq}"), "overlap_at_k": overlap as f64 / (nq * k) as f64,
            "max_score_diff": max_sdiff, "workspace_mb": idx.mem().2}));
    }
    let _ = wm;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("parity") => parity(&args),
        Some("bench") => bench(&args),
        Some("embed") => cmd_embed(&args),
        Some("prep") => cmd_prep(&args),
        Some("tokenize") => cmd_tokenize(&args),
        Some("search") => cmd_search(&args),
        Some("search-bench") => cmd_search_bench(&args),
        Some("export") => {
            let md = arg(&args, "--model-dir").ok_or(anyhow!("--model-dir"))?;
            let out = arg(&args, "--out").ok_or(anyhow!("--out"))?;
            println!("{}", export::export(&md, &out)?);
            Ok(())
        }
        Some("serve") => server::serve(
            &arg(&args, "--assets").ok_or(anyhow!("--assets"))?,
            arg(&args, "--port").map(|s| s.parse()).transpose()?.unwrap_or(8080),
            arg(&args, "--threads").map(|s| s.parse()).transpose()?,
            args.iter().any(|a| a == "--fp32"),
        ),
        _ => {
            eprintln!("usage: clipfast-rs parity|bench|embed|prep|tokenize|export|serve ... (see the header of src/main.rs)");
            let _ = HashMap::<u8, u8>::new();
            Ok(())
        }
    }
}