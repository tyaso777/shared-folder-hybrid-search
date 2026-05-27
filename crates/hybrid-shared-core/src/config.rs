use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::embedding::EmbeddingConfigOverride;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SharedSearchConfig {
    pub shared_root: Option<PathBuf>,
    pub dataset_id: Option<String>,
    // Backward-compatible alias. Prefer dataset_id in new config files.
    pub dataset: Option<String>,
    pub indexes_root: Option<PathBuf>,
    pub embedding_model: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
    pub ort_dll: Option<PathBuf>,
    pub embedding_dim: Option<usize>,
    pub max_input_tokens: Option<usize>,
    pub embedding_model_id: Option<String>,
    pub query_prefix: Option<String>,
    pub document_prefix: Option<String>,
    pub preload_model_to_memory: Option<bool>,
    pub chunk_mode: Option<String>,
    pub chunk_size: Option<usize>,
    pub chunk_overlap: Option<usize>,
    pub poll_seconds: Option<u64>,
    pub done_ttl_secs: Option<u64>,
    pub failed_ttl_secs: Option<u64>,
    pub cleanup_interval_secs: Option<u64>,
    pub no_open: Option<bool>,
    pub keep_responses: Option<bool>,
    pub default_top_k: Option<usize>,
    pub request_timeout_secs: Option<u64>,
    pub browser_shutdown_secs: Option<u64>,
    pub search_poll_interval_ms: Option<u64>,
    pub client_port: Option<u16>,
}

impl SharedSearchConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&text)?)
    }

    pub fn load_resolved(path: &Path) -> anyhow::Result<Self> {
        let mut config = Self::load(path)?;
        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
        config.shared_root = config
            .shared_root
            .map(|path| resolve_relative(base_dir, path));
        config.indexes_root = config
            .indexes_root
            .map(|path| resolve_relative(base_dir, path));
        config.embedding_model = config
            .embedding_model
            .map(|path| resolve_relative(base_dir, path));
        config.tokenizer = config
            .tokenizer
            .map(|path| resolve_relative(base_dir, path));
        config.ort_dll = config.ort_dll.map(|path| resolve_relative(base_dir, path));
        config.apply_env_overrides();
        Ok(config)
    }

    pub fn with_env_overrides(mut self) -> Self {
        self.apply_env_overrides();
        self
    }

    fn apply_env_overrides(&mut self) {
        if let Some(value) = env_path("SHARED_SEARCH_EMBEDDING_MODEL") {
            self.embedding_model = Some(value);
        }
        if let Some(value) = env_path("SHARED_SEARCH_TOKENIZER") {
            self.tokenizer = Some(value);
        }
        if let Some(value) = env_path("SHARED_SEARCH_ORT_DLL") {
            self.ort_dll = Some(value);
        }
    }

    pub fn embedding_override(&self) -> EmbeddingConfigOverride {
        EmbeddingConfigOverride {
            model_path: self.embedding_model.clone(),
            tokenizer_path: self.tokenizer.clone(),
            runtime_library_path: self.ort_dll.clone(),
            dimension: self.embedding_dim,
            max_input_tokens: self.max_input_tokens,
            model_id: self.embedding_model_id.clone(),
            query_prefix: self.query_prefix.clone(),
            document_prefix: self.document_prefix.clone(),
            preload_model_to_memory: self.preload_model_to_memory,
        }
    }

    pub fn dataset_id(&self) -> Option<String> {
        self.dataset_id.clone().or_else(|| self.dataset.clone())
    }
}

fn resolve_relative(base_dir: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    }
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

pub fn default_config_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    exe.parent().map(|dir| dir.join("shared-search.toml"))
}
