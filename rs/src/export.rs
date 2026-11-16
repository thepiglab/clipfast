//! Weight export without PyTorch: read the open_clip checkpoint (`open_clip_pytorch_model.bin`, a PyTorch zip
//! archive: `*/data.pkl` pickle + `*/data/<key>` raw storages) and write the runtime's safetensors files
//! (`weights_f32.safetensors`, `weights_bf16.safetensors`), plus `config.json` and `tokenizer.json`.
//! The tensor transforms mirror the runtime's layout: BERT q/k/v fused, patch conv as a [768, 768] GEMM weight,
//! cls + pos[0] and position + token-type-0 embeddings folded (summed in fp32, then rounded once to bf16).
use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashMap;
use std::io::Read;

// ---------------- minimal pickle VM for torch.save's state_dict ----------------
#[derive(Clone, Debug)]
enum V {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<V>),
    List(Vec<V>),
    Dict(Vec<(V, V)>),
    Global(String, String),
    Storage { key: String, dtype: String },
    Tensor(TensorRef),
    Mark,
    Other,
}

#[derive(Clone, Debug)]
pub struct TensorRef {
    pub key: String,
    pub dtype: String,
    pub offset: usize,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
}

fn unpickle(data: &[u8]) -> Result<V> {
    let mut st: Vec<V> = Vec::new();
    let mut memo: HashMap<u64, V> = HashMap::new();
    let mut i = 0usize;
    let rd = |i: &mut usize, n: usize| -> Result<&[u8]> {
        let s = data.get(*i..*i + n).ok_or_else(|| anyhow!("pickle truncated"))?;
        *i += n;
        Ok(s)
    };
    let pop_mark = |st: &mut Vec<V>| -> Result<Vec<V>> {
        let pos = st.iter().rposition(|v| matches!(v, V::Mark)).ok_or_else(|| anyhow!("no MARK"))?;
        let items = st.split_off(pos + 1);
        st.pop();
        Ok(items)
    };
    loop {
        let op = *data.get(i).ok_or_else(|| anyhow!("pickle ended without STOP"))?;
        i += 1;
        match op {
            0x80 => { rd(&mut i, 1)?; }                                          // PROTO
            0x95 => { rd(&mut i, 8)?; }                                          // FRAME
            b'.' => return st.pop().ok_or_else(|| anyhow!("empty stack")),        // STOP
            b'}' => st.push(V::Dict(vec![])),                                    // EMPTY_DICT
            b']' => st.push(V::List(vec![])),                                    // EMPTY_LIST
            b')' => st.push(V::Tuple(vec![])),                                   // EMPTY_TUPLE
            b'(' => st.push(V::Mark),                                            // MARK
            b'N' => st.push(V::None),
            0x88 => st.push(V::Bool(true)),
            0x89 => st.push(V::Bool(false)),
            b'K' => { let b = rd(&mut i, 1)?[0]; st.push(V::Int(b as i64)); }     // BININT1
            b'M' => { let b = rd(&mut i, 2)?; st.push(V::Int(u16::from_le_bytes([b[0], b[1]]) as i64)); }
            b'J' => { let b = rd(&mut i, 4)?; st.push(V::Int(i32::from_le_bytes(b.try_into()?) as i64)); }
            0x8a => {                                                             // LONG1
                let n = rd(&mut i, 1)?[0] as usize;
                let b = rd(&mut i, n)?;
                let mut v: i64 = 0;
                for (k, &x) in b.iter().enumerate().take(8) { v |= (x as i64) << (8 * k); }
                if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 { v -= 1i64 << (8 * n); }
                st.push(V::Int(v));
            }
            b'G' => { let b = rd(&mut i, 8)?; st.push(V::Float(f64::from_be_bytes(b.try_into()?))); }
            b'X' => {                                                             // BINUNICODE
                let n = u32::from_le_bytes(rd(&mut i, 4)?.try_into()?) as usize;
                st.push(V::Str(String::from_utf8(rd(&mut i, n)?.to_vec())?));
            }
            0x8c => {                                                             // SHORT_BINUNICODE
                let n = rd(&mut i, 1)?[0] as usize;
                st.push(V::Str(String::from_utf8(rd(&mut i, n)?.to_vec())?));
            }
            b'c' => {                                                             // GLOBAL "module\nname\n"
                let start = i;
                let nl1 = data[start..].iter().position(|&c| c == b'\n').ok_or_else(|| anyhow!("GLOBAL"))? + start;
                let nl2 = data[nl1 + 1..].iter().position(|&c| c == b'\n').ok_or_else(|| anyhow!("GLOBAL"))? + nl1 + 1;
                let m = String::from_utf8(data[start..nl1].to_vec())?;
                let n = String::from_utf8(data[nl1 + 1..nl2].to_vec())?;
                i = nl2 + 1;
                st.push(V::Global(m, n));
            }
            0x93 => {                                                             // STACK_GLOBAL
                let n = st.pop(); let m = st.pop();
                match (m, n) { (Some(V::Str(m)), Some(V::Str(n))) => st.push(V::Global(m, n)), _ => bail!("STACK_GLOBAL") }
            }
            b'q' => { let k = rd(&mut i, 1)?[0] as u64; memo.insert(k, st.last().cloned().unwrap_or(V::None)); }
            b'r' => { let k = u32::from_le_bytes(rd(&mut i, 4)?.try_into()?) as u64; memo.insert(k, st.last().cloned().unwrap_or(V::None)); }
            0x94 => { let k = memo.len() as u64; memo.insert(k, st.last().cloned().unwrap_or(V::None)); } // MEMOIZE
            b'h' => { let k = rd(&mut i, 1)?[0] as u64; st.push(memo.get(&k).cloned().ok_or_else(|| anyhow!("memo {k}"))?); }
            b'j' => { let k = u32::from_le_bytes(rd(&mut i, 4)?.try_into()?) as u64; st.push(memo.get(&k).cloned().ok_or_else(|| anyhow!("memo {k}"))?); }
            b't' => { let items = pop_mark(&mut st)?; st.push(V::Tuple(items)); }
            0x85 => { let a = st.pop().unwrap(); st.push(V::Tuple(vec![a])); }
            0x86 => { let b = st.pop().unwrap(); let a = st.pop().unwrap(); st.push(V::Tuple(vec![a, b])); }
            0x87 => { let c = st.pop().unwrap(); let b = st.pop().unwrap(); let a = st.pop().unwrap(); st.push(V::Tuple(vec![a, b, c])); }
            b'a' => { let x = st.pop().unwrap(); if let Some(V::List(l)) = st.last_mut() { l.push(x) } }
            b'e' => { let items = pop_mark(&mut st)?; if let Some(V::List(l)) = st.last_mut() { l.extend(items) } }
            b's' => { let v = st.pop().unwrap(); let k = st.pop().unwrap(); if let Some(V::Dict(d)) = st.last_mut() { d.push((k, v)) } }
            b'u' => {
                let items = pop_mark(&mut st)?;
                if let Some(V::Dict(d)) = st.last_mut() {
                    for kv in items.chunks(2) { if kv.len() == 2 { d.push((kv[0].clone(), kv[1].clone())) } }
                }
            }
            b'Q' => {                                                             // BINPERSID: ('storage', cls, key, loc, numel)
                let pid = st.pop().unwrap();
                if let V::Tuple(t) = pid {
                    let dtype = match t.get(1) { Some(V::Global(_, n)) => n.clone(), _ => String::new() };
                    let key = match t.get(2) { Some(V::Str(s)) => s.clone(), _ => bail!("storage key") };
                    st.push(V::Storage { key, dtype });
                } else { bail!("BINPERSID") }
            }
            b'R' => {                                                             // REDUCE
                let args = st.pop().unwrap();
                let f = st.pop().unwrap();
                let v = match (&f, &args) {
                    (V::Global(m, n), V::Tuple(a)) if m == "torch._utils" && n == "_rebuild_tensor_v2" => {
                        let (key, dtype) = match &a[0] { V::Storage { key, dtype } => (key.clone(), dtype.clone()), _ => bail!("tensor storage") };
                        let offset = match &a[1] { V::Int(x) => *x as usize, _ => 0 };
                        let ints = |v: &V| -> Vec<usize> { match v { V::Tuple(t) => t.iter().filter_map(|x| if let V::Int(i) = x { Some(*i as usize) } else { None }).collect(), _ => vec![] } };
                        V::Tensor(TensorRef { key, dtype, offset, shape: ints(&a[2]), stride: ints(&a[3]) })
                    }
                    (V::Global(_, n), _) if n == "OrderedDict" => V::Dict(vec![]),
                    _ => V::Other,
                };
                st.push(v);
            }
            b'b' => { st.pop(); }                                                 // BUILD (state ignored)
            b'0' => { st.pop(); }                                                 // POP
            other => bail!("unsupported pickle opcode 0x{other:02x} at {}", i - 1),
        }
    }
}

/// Load every tensor of a torch.save zip checkpoint as contiguous f32 (supports float32/float16/bfloat16 storages).
pub fn load_torch_zip(path: &str) -> Result<HashMap<String, (Vec<usize>, Vec<f32>)>> {
    let f = std::fs::File::open(path).with_context(|| path.to_string())?;
    let mut z = zip::ZipArchive::new(f).context("not a torch zip checkpoint")?;
    let pkl_name = (0..z.len()).map(|k| z.by_index(k).map(|e| e.name().to_string())).collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter().find(|n| n.ends_with("data.pkl")).ok_or_else(|| anyhow!("no data.pkl"))?;
    let prefix = pkl_name.trim_end_matches("data.pkl").to_string();
    let mut pkl = vec![];
    z.by_name(&pkl_name)?.read_to_end(&mut pkl)?;
    let root = unpickle(&pkl)?;
    let entries = match root { V::Dict(d) => d, _ => bail!("checkpoint root is not a dict") };
    let mut out = HashMap::new();
    let mut cache: HashMap<String, Vec<u8>> = HashMap::new();
    for (k, v) in entries {
        let (V::Str(name), V::Tensor(t)) = (k, v) else { continue };
        if !cache.contains_key(&t.key) {
            let mut b = vec![];
            z.by_name(&format!("{prefix}data/{}", t.key))?.read_to_end(&mut b)?;
            cache.insert(t.key.clone(), b);
        }
        let raw = &cache[&t.key];
        let (es, conv): (usize, fn(&[u8]) -> f32) = match t.dtype.as_str() {
            "FloatStorage" => (4, |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            "HalfStorage" => (2, |b| half_to_f32(u16::from_le_bytes([b[0], b[1]]))),
            "BFloat16Storage" => (2, |b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)),
            "LongStorage" | "IntStorage" => continue,                             // e.g. position_ids buffers
            d => bail!("{name}: unsupported storage {d}"),
        };
        let numel: usize = t.shape.iter().product::<usize>().max(1);
        let mut vals = vec![0f32; numel];
        // gather with strides (state_dict tensors are usually contiguous, but be general)
        let nd = t.shape.len();
        let mut idx = vec![0usize; nd];
        for o in vals.iter_mut() {
            let mut off = t.offset;
            for d in 0..nd { off += idx[d] * t.stride[d]; }
            *o = conv(&raw[off * es..off * es + es]);
            for d in (0..nd).rev() {
                idx[d] += 1;
                if idx[d] < t.shape[d] { break; }
                idx[d] = 0;
            }
        }
        out.insert(name, (t.shape.clone(), vals));
    }
    Ok(out)
}

fn half_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 { s } else {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 { m2 <<= 1; e2 -= 1; }
            s | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 { s | 0x7f80_0000 | (m << 13) } else { s | ((e + 127 - 15) << 23) | (m << 13) };
    f32::from_bits(bits)
}

/// f32 -> bf16 bits, round to nearest even (as torch's .to(torch.bfloat16)); NaN kept quiet.
pub fn f32_to_bf16(x: f32) -> u16 {
    let u = x.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 { return ((u >> 16) | 0x40) as u16; }
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

/// Build the runtime's tensors from an open_clip BiomedCLIP state dict.
pub fn runtime_tensors(sd: &HashMap<String, (Vec<usize>, Vec<f32>)>) -> Result<Vec<(String, Vec<usize>, Vec<f32>)>> {
    let g = |k: &str| -> Result<&(Vec<usize>, Vec<f32>)> { sd.get(k).ok_or_else(|| anyhow!("checkpoint lacks {k}")) };
    let mut out: Vec<(String, Vec<usize>, Vec<f32>)> = vec![];
    let mut put = |n: &str, shape: Vec<usize>, v: Vec<f32>| out.push((n.to_string(), shape, v));
    let (_, pw) = g("visual.trunk.patch_embed.proj.weight")?;
    put("v.patch.w", vec![768, 768], pw.clone());
    put("v.patch.b", vec![768], g("visual.trunk.patch_embed.proj.bias")?.1.clone());
    let cls = &g("visual.trunk.cls_token")?.1;
    let pos = &g("visual.trunk.pos_embed")?.1; // [1, 197, 768]
    put("v.cls_pos0", vec![768], (0..768).map(|c| cls[c] + pos[c]).collect());
    put("v.pos", vec![196, 768], pos[768..].to_vec());
    for i in 0..12 {
        let p = format!("visual.trunk.blocks.{i}.");
        for (dst, src) in [("ln1.w", "norm1.weight"), ("ln1.b", "norm1.bias"), ("qkv.w", "attn.qkv.weight"), ("qkv.b", "attn.qkv.bias"),
                           ("o.w", "attn.proj.weight"), ("o.b", "attn.proj.bias"), ("ln2.w", "norm2.weight"), ("ln2.b", "norm2.bias"),
                           ("fc1.w", "mlp.fc1.weight"), ("fc1.b", "mlp.fc1.bias"), ("fc2.w", "mlp.fc2.weight"), ("fc2.b", "mlp.fc2.bias")] {
            let (s, v) = g(&(p.clone() + src))?;
            put(&format!("v.{i}.{dst}"), s.clone(), v.clone());
        }
    }
    put("v.ln.w", vec![768], g("visual.trunk.norm.weight")?.1.clone());
    put("v.ln.b", vec![768], g("visual.trunk.norm.bias")?.1.clone());
    let (ps, pv) = g("visual.head.proj.weight")?;
    put("v.proj.w", ps.clone(), pv.clone());
    let e = "text.transformer.embeddings.";
    let (ws, wv) = g(&format!("{e}word_embeddings.weight"))?;
    put("t.word", ws.clone(), wv.clone());
    let (pos_s, posv) = g(&format!("{e}position_embeddings.weight"))?;
    let typ = &g(&format!("{e}token_type_embeddings.weight"))?.1;
    put("t.pos_type", pos_s.clone(), posv.iter().enumerate().map(|(k, x)| x + typ[k % 768]).collect());
    put("t.ln.w", vec![768], g(&format!("{e}LayerNorm.weight"))?.1.clone());
    put("t.ln.b", vec![768], g(&format!("{e}LayerNorm.bias"))?.1.clone());
    for i in 0..12 {
        let p = format!("text.transformer.encoder.layer.{i}.");
        let cat = |a: &str| -> Result<Vec<f32>> {
            let mut v = g(&format!("{p}attention.self.query.{a}"))?.1.clone();
            v.extend_from_slice(&g(&format!("{p}attention.self.key.{a}"))?.1);
            v.extend_from_slice(&g(&format!("{p}attention.self.value.{a}"))?.1);
            Ok(v)
        };
        put(&format!("t.{i}.qkv.w"), vec![2304, 768], cat("weight")?);
        put(&format!("t.{i}.qkv.b"), vec![2304], cat("bias")?);
        for (dst, src) in [("o.w", "attention.output.dense.weight"), ("o.b", "attention.output.dense.bias"),
                           ("ln1.w", "attention.output.LayerNorm.weight"), ("ln1.b", "attention.output.LayerNorm.bias"),
                           ("fc1.w", "intermediate.dense.weight"), ("fc1.b", "intermediate.dense.bias"),
                           ("fc2.w", "output.dense.weight"), ("fc2.b", "output.dense.bias"),
                           ("ln2.w", "output.LayerNorm.weight"), ("ln2.b", "output.LayerNorm.bias")] {
            let (s, v) = g(&(p.clone() + src))?;
            put(&format!("t.{i}.{dst}"), s.clone(), v.clone());
        }
    }
    put("t.p1.w", g("text.proj.0.weight")?.0.clone(), g("text.proj.0.weight")?.1.clone());
    put("t.p2.w", g("text.proj.2.weight")?.0.clone(), g("text.proj.2.weight")?.1.clone());
    put("logit_scale", vec![1], vec![g("logit_scale")?.1[0]]);
    Ok(out)
}

fn write_safetensors(path: &str, tensors: &[(String, Vec<usize>, Vec<f32>)], bf16: bool) -> Result<()> {
    use std::io::Write;
    let mut sorted: Vec<&(String, Vec<usize>, Vec<f32>)> = tensors.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let es = if bf16 { 2 } else { 4 };
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for (n, s, v) in &sorted {
        let len = v.len() * es;
        header.insert(n.clone(), serde_json::json!({"dtype": if bf16 {"BF16"} else {"F32"}, "shape": s, "data_offsets": [off, off + len]}));
        off += len;
    }
    let mut h = serde_json::to_string(&serde_json::Value::Object(header))?;
    while h.len() % 8 != 0 { h.push(' '); }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&(h.len() as u64).to_le_bytes())?;
    f.write_all(h.as_bytes())?;
    for (_, _, v) in &sorted {
        for x in v.iter() {
            if bf16 { f.write_all(&f32_to_bf16(*x).to_le_bytes())?; } else { f.write_all(&x.to_le_bytes())?; }
        }
    }
    Ok(())
}

/// `clipfast-rs export --model-dir DIR --out DIR`
pub fn export(model_dir: &str, out: &str) -> Result<serde_json::Value> {
    std::fs::create_dir_all(out)?;
    let sd = load_torch_zip(&format!("{model_dir}/open_clip_pytorch_model.bin"))?;
    let t = runtime_tensors(&sd)?;
    write_safetensors(&format!("{out}/weights_f32.safetensors"), &t, false)?;
    write_safetensors(&format!("{out}/weights_bf16.safetensors"), &t, true)?;
    std::fs::copy(format!("{model_dir}/tokenizer.json"), format!("{out}/tokenizer.json"))?;
    let ls = g_scalar(&sd, "logit_scale")?.exp();
    let cfg = serde_json::json!({"width": 768, "heads": 12, "layers": 12, "mlp": 3072, "embed": 512, "text_proj_hidden": 640,
        "img": 224, "patch": 16, "tokens": 197, "vocab": sd["text.transformer.embeddings.word_embeddings.weight"].0[0],
        "max_pos": sd["text.transformer.embeddings.position_embeddings.weight"].0[0], "v_eps": 1e-6, "t_eps": 1e-12,
        "pad_id": 0, "cls_id": 2, "sep_id": 3, "logit_scale": ls});
    std::fs::write(format!("{out}/config.json"), serde_json::to_string_pretty(&cfg)?)?;
    Ok(serde_json::json!({"kind": "export", "tensors": t.len(), "out": out}))
}

fn g_scalar(sd: &HashMap<String, (Vec<usize>, Vec<f32>)>, k: &str) -> Result<f32> {
    Ok(sd.get(k).ok_or_else(|| anyhow!("missing {k}"))?.1[0])
}
