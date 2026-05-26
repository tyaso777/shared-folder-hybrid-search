use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ndarray::Array2;
use ort::{session::Session, value::Tensor};
use serde::{Deserialize, Serialize};
use tokenizers::{Encoding, Tokenizer, TruncationParams};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub runtime_library_path: PathBuf,
    pub dimension: usize,
    #[serde(default = "default_max_input_tokens")]
    pub max_input_tokens: usize,
    #[serde(default = "default_model_id")]
    pub model_id: String,
    #[serde(default = "default_query_prefix")]
    pub query_prefix: String,
    #[serde(default = "default_document_prefix")]
    pub document_prefix: String,
    #[serde(default)]
    pub preload_model_to_memory: bool,
}

pub struct OnnxEmbedder {
    cfg: EmbeddingConfig,
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    pad_id: i64,
}

static ORT_RUNTIME_PATH: OnceLock<PathBuf> = OnceLock::new();

impl OnnxEmbedder {
    pub fn new(cfg: EmbeddingConfig) -> anyhow::Result<Self> {
        if cfg.dimension == 0 {
            anyhow::bail!("embedding dimension must be greater than zero");
        }
        ensure_ort_initialized(&cfg.runtime_library_path)?;
        let session = if cfg.preload_model_to_memory {
            let bytes = fs::read(&cfg.model_path)?;
            Session::builder()?.commit_from_memory(&bytes)?
        } else {
            Session::builder()?.commit_from_file(&cfg.model_path)?
        };
        let tokenizer = Tokenizer::from_file(&cfg.tokenizer_path)
            .map_err(|err| anyhow::anyhow!("load tokenizer failed: {err}"))?;
        let pad_id = tokenizer
            .token_to_id("<pad>")
            .ok_or_else(|| anyhow::anyhow!("tokenizer does not declare <pad> token"))?
            as i64;
        Ok(Self {
            cfg,
            session: Mutex::new(session),
            tokenizer,
            pad_id,
        })
    }

    pub fn dimension(&self) -> usize {
        self.cfg.dimension
    }

    pub fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let mut batch = self.embed_batch(&[text])?;
        batch
            .pop()
            .ok_or_else(|| anyhow::anyhow!("missing embedding output"))
    }

    pub fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.embed_batch_raw(texts)
    }

    pub fn embed_query(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let prefixed = format!("{}{}", self.cfg.query_prefix, text);
        self.embed(&prefixed)
    }

    pub fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let prefixed = texts
            .iter()
            .map(|text| format!("{}{}", self.cfg.document_prefix, text))
            .collect::<Vec<_>>();
        let refs = prefixed.iter().map(String::as_str).collect::<Vec<_>>();
        self.embed_batch_raw(&refs)
    }

    fn embed_batch_raw(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self.prepare_encodings(texts)?;
        let (input_ids, attention_mask, attention_rows) = self.build_inputs(&encodings)?;
        let mut session = self.session.lock().expect("ONNX session mutex poisoned");
        let outputs = session.run(ort::inputs![input_ids, attention_mask])?;
        let output = &outputs[0];
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        if shape.len() != 3 {
            anyhow::bail!("model output must be rank-3 [batch, seq_len, hidden], got {shape:?}");
        }
        let batch = shape[0] as usize;
        let seq_len = shape[1] as usize;
        let hidden = shape[2] as usize;
        if batch != attention_rows.len() {
            anyhow::bail!("model batch size mismatch");
        }
        let vectors = mean_pool(data, &attention_rows, seq_len, hidden);
        if vectors.iter().any(|v| v.len() != self.cfg.dimension) {
            anyhow::bail!("embedding dimension mismatch");
        }
        Ok(vectors)
    }

    fn prepare_encodings(&self, texts: &[&str]) -> anyhow::Result<Vec<Encoding>> {
        let mut tokenizer = self.tokenizer.clone();
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: self.cfg.max_input_tokens,
                ..Default::default()
            }))
            .map_err(|err| anyhow::anyhow!("set tokenizer truncation failed: {err}"))?;
        let encodings = texts
            .iter()
            .map(|text| tokenizer.encode(*text, true))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| anyhow::anyhow!("tokenize failed: {err}"))?;
        Ok(encodings)
    }

    fn build_inputs(
        &self,
        encodings: &[Encoding],
    ) -> anyhow::Result<(Tensor<i64>, Tensor<i64>, Vec<Vec<i64>>)> {
        let batch = encodings.len();
        let seq_len = encodings.iter().map(Encoding::len).max().unwrap_or(0);
        let mut input_ids = Array2::<i64>::zeros((batch, seq_len));
        let mut attention_mask = Array2::<i64>::zeros((batch, seq_len));
        let mut rows = Vec::with_capacity(batch);
        for (row, encoding) in encodings.iter().enumerate() {
            for (col, (&id, &mask)) in encoding
                .get_ids()
                .iter()
                .zip(encoding.get_attention_mask())
                .enumerate()
            {
                input_ids[(row, col)] = id as i64;
                attention_mask[(row, col)] = mask as i64;
            }
            for col in encoding.len()..seq_len {
                input_ids[(row, col)] = self.pad_id;
                attention_mask[(row, col)] = 0;
            }
            rows.push((0..seq_len).map(|col| attention_mask[(row, col)]).collect());
        }
        Ok((
            Tensor::from_array(input_ids)?,
            Tensor::from_array(attention_mask)?,
            rows,
        ))
    }
}

pub fn resolve_config_paths(cfg: &mut EmbeddingConfig, base: &Path) {
    let base = if base.is_absolute() {
        base.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(base)
    };
    if cfg.model_path.is_relative() {
        cfg.model_path = base.join(&cfg.model_path);
    }
    if cfg.tokenizer_path.is_relative() {
        cfg.tokenizer_path = base.join(&cfg.tokenizer_path);
    }
    if cfg.runtime_library_path.is_relative() {
        cfg.runtime_library_path = base.join(&cfg.runtime_library_path);
    }
}

fn ensure_ort_initialized(path: &Path) -> anyhow::Result<()> {
    let canonical = path.canonicalize()?;
    if let Some(existing) = ORT_RUNTIME_PATH.get() {
        if existing != &canonical {
            anyhow::bail!(
                "ONNX Runtime already initialized with `{}`; cannot switch to `{}`",
                existing.display(),
                canonical.display()
            );
        }
    } else {
        let _ = ORT_RUNTIME_PATH.set(canonical.clone());
    }
    ort::init_from(canonical.to_string_lossy().to_string())
        .with_name("shared-folder-hybrid-search")
        .commit()?;
    Ok(())
}

fn mean_pool(data: &[f32], masks: &[Vec<i64>], seq_len: usize, hidden: usize) -> Vec<Vec<f32>> {
    let mut output = Vec::with_capacity(masks.len());
    for (batch, mask) in masks.iter().enumerate() {
        let mut sum = vec![0.0; hidden];
        let mut count = 0.0f32;
        for token in 0..seq_len {
            if mask[token] == 1 {
                let base = (batch * seq_len + token) * hidden;
                for h in 0..hidden {
                    sum[h] += data[base + h];
                }
                count += 1.0;
            }
        }
        if count > 0.0 {
            for value in &mut sum {
                *value /= count;
            }
        }
        output.push(sum);
    }
    output
}

fn default_max_input_tokens() -> usize {
    8192
}

fn default_model_id() -> String {
    "ruri-v3-onnx".to_string()
}

fn default_query_prefix() -> String {
    "検索クエリ: ".to_string()
}

fn default_document_prefix() -> String {
    "検索文書: ".to_string()
}
