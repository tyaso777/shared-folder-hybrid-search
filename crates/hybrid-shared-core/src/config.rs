use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SharedSearchConfig {
    pub shared_root: Option<PathBuf>,
    pub dataset: Option<String>,
    pub indexes_root: Option<PathBuf>,
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
        Ok(config)
    }
}

fn resolve_relative(base_dir: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    }
}

pub fn default_config_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    exe.parent().map(|dir| dir.join("shared-search.toml"))
}
