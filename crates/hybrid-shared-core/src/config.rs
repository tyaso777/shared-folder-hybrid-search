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
}

impl SharedSearchConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&text)?)
    }
}

pub fn default_config_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    exe.parent().map(|dir| dir.join("shared-search.toml"))
}
