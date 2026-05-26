use std::fs;
use std::path::{Path, PathBuf};

use serde::{de::DeserializeOwned, Serialize};

pub fn ensure_layout(root: &Path) -> std::io::Result<()> {
    for dir in [
        root.join("requests").join("pending"),
        root.join("requests").join("processing"),
        root.join("requests").join("done"),
        root.join("requests").join("failed"),
        root.join("responses"),
    ] {
        fs::create_dir_all(dir)?;
    }
    Ok(())
}

pub fn atomic_write_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = tmp_path(path);
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(&tmp, bytes)?;
    fs::rename(tmp, path)?;
    Ok(())
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    let bytes = fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub fn request_path(root: &Path, request_id: &str) -> PathBuf {
    root.join("requests")
        .join("pending")
        .join(format!("{request_id}.request.json"))
}

pub fn response_path(root: &Path, client_id: &str, request_id: &str) -> PathBuf {
    root.join("responses")
        .join(client_id)
        .join(format!("{request_id}.response.json"))
}

pub fn tmp_path(path: &Path) -> PathBuf {
    let mut tmp = path.to_path_buf();
    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("file");
    tmp.set_file_name(format!("{file_name}.tmp"));
    tmp
}
