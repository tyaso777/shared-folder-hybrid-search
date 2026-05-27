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

#[derive(Debug, Clone, Default)]
pub struct EmbeddingConfigOverride {
    pub model_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub runtime_library_path: Option<PathBuf>,
    pub dimension: Option<usize>,
    pub max_input_tokens: Option<usize>,
    pub model_id: Option<String>,
    pub query_prefix: Option<String>,
    pub document_prefix: Option<String>,
    pub preload_model_to_memory: Option<bool>,
}

impl EmbeddingConfigOverride {
    pub fn is_empty(&self) -> bool {
        self.model_path.is_none()
            && self.tokenizer_path.is_none()
            && self.runtime_library_path.is_none()
            && self.dimension.is_none()
            && self.max_input_tokens.is_none()
            && self.model_id.is_none()
            && self.query_prefix.is_none()
            && self.document_prefix.is_none()
            && self.preload_model_to_memory.is_none()
    }

    pub fn apply_to(&self, cfg: &mut EmbeddingConfig) {
        if let Some(value) = &self.model_path {
            cfg.model_path = value.clone();
        }
        if let Some(value) = &self.tokenizer_path {
            cfg.tokenizer_path = value.clone();
        }
        if let Some(value) = &self.runtime_library_path {
            cfg.runtime_library_path = value.clone();
        }
        if let Some(value) = self.dimension {
            cfg.dimension = value;
        }
        if let Some(value) = self.max_input_tokens {
            cfg.max_input_tokens = value;
        }
        if let Some(value) = &self.model_id {
            cfg.model_id = value.clone();
        }
        if let Some(value) = &self.query_prefix {
            cfg.query_prefix = value.clone();
        }
        if let Some(value) = &self.document_prefix {
            cfg.document_prefix = value.clone();
        }
        if let Some(value) = self.preload_model_to_memory {
            cfg.preload_model_to_memory = value;
        }
    }

    pub fn to_config(&self) -> anyhow::Result<Option<EmbeddingConfig>> {
        let any_path = self.model_path.is_some()
            || self.tokenizer_path.is_some()
            || self.runtime_library_path.is_some();
        if !any_path {
            return Ok(None);
        }
        let (Some(model_path), Some(tokenizer_path), Some(runtime_library_path)) = (
            self.model_path.clone(),
            self.tokenizer_path.clone(),
            self.runtime_library_path.clone(),
        ) else {
            anyhow::bail!("embedding_model, tokenizer, and ort_dll must be configured together");
        };
        Ok(Some(EmbeddingConfig {
            model_path,
            tokenizer_path,
            runtime_library_path,
            dimension: self.dimension.unwrap_or(768),
            max_input_tokens: self
                .max_input_tokens
                .unwrap_or_else(default_max_input_tokens),
            model_id: self.model_id.clone().unwrap_or_else(default_model_id),
            query_prefix: self
                .query_prefix
                .clone()
                .unwrap_or_else(|| "検索クエリ: ".to_string()),
            document_prefix: self
                .document_prefix
                .clone()
                .unwrap_or_else(|| "検索文書: ".to_string()),
            preload_model_to_memory: self.preload_model_to_memory.unwrap_or(false),
        }))
    }
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
