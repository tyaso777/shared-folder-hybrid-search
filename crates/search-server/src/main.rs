use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use hybrid_shared_core::config::{default_config_path_for, SharedSearchConfig};
use hybrid_shared_core::embedding::EmbeddingConfigOverride;
use hybrid_shared_core::index::load_current_index_with_embedding_override;
use hybrid_shared_core::protocol::{Request, Response, ResponseError, ResponseOk};
use hybrid_shared_core::shared_folder::{
    atomic_write_json, ensure_layout, read_json, response_path,
};

#[derive(Debug, Parser)]
#[command(author, version, about = "Shared-folder search server")]
struct Args {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    shared_root: Option<PathBuf>,
    #[arg(long)]
    indexes_root: Option<PathBuf>,
    #[arg(long)]
    poll_seconds: Option<u64>,
    #[arg(long)]
    done_ttl_secs: Option<u64>,
    #[arg(long)]
    failed_ttl_secs: Option<u64>,
    #[arg(long)]
    cleanup_interval_secs: Option<u64>,
}

#[derive(Debug)]
struct ResolvedArgs {
    shared_root: PathBuf,
    indexes_root: PathBuf,
    poll_seconds: u64,
    done_ttl_secs: u64,
    failed_ttl_secs: u64,
    cleanup_interval_secs: u64,
    embedding_override: EmbeddingConfigOverride,
}

fn main() -> anyhow::Result<()> {
    let args = resolve_args(Args::parse())?;
    ensure_layout(&args.shared_root)?;
    cleanup_old_files(&args.shared_root, args.done_ttl_secs, args.failed_ttl_secs)?;
    let mut last_cleanup = Instant::now();
    println!("watching {}", args.shared_root.display());
    println!("using indexes_root {}", args.indexes_root.display());
    loop {
        process_once(
            &args.shared_root,
            &args.indexes_root,
            &args.embedding_override,
        )?;
        if last_cleanup.elapsed() >= Duration::from_secs(args.cleanup_interval_secs) {
            cleanup_old_files(&args.shared_root, args.done_ttl_secs, args.failed_ttl_secs)?;
            last_cleanup = Instant::now();
        }
        thread::sleep(Duration::from_secs(args.poll_seconds));
    }
}

fn resolve_args(args: Args) -> anyhow::Result<ResolvedArgs> {
    let config_path = args
        .config
        .or_else(|| default_config_path_for("server.toml"));
    let config = match config_path {
        Some(path) if path.exists() => SharedSearchConfig::load_resolved(&path)?,
        _ => SharedSearchConfig::default().with_env_overrides(),
    };
    let embedding_override = config.embedding_override();
    Ok(ResolvedArgs {
        shared_root: args
            .shared_root
            .or(config.shared_root)
            .unwrap_or_else(|| PathBuf::from("shared_demo")),
        indexes_root: args
            .indexes_root
            .or(config.indexes_root)
            .unwrap_or_else(|| PathBuf::from("indexes")),
        poll_seconds: args.poll_seconds.or(config.poll_seconds).unwrap_or(2),
        done_ttl_secs: args.done_ttl_secs.or(config.done_ttl_secs).unwrap_or(600),
        failed_ttl_secs: args
            .failed_ttl_secs
            .or(config.failed_ttl_secs)
            .unwrap_or(86_400),
        cleanup_interval_secs: args
            .cleanup_interval_secs
            .or(config.cleanup_interval_secs)
            .unwrap_or(60),
        embedding_override,
    })
}

fn process_once(
    shared_root: &Path,
    indexes_root: &Path,
    embedding_override: &EmbeddingConfigOverride,
) -> anyhow::Result<()> {
    let pending = shared_root.join("requests").join("pending");
    for entry in fs::read_dir(pending)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let Some(file_name) = path
            .file_name()
            .and_then(|s| s.to_str())
            .map(str::to_string)
        else {
            continue;
        };
        if !file_name.ends_with(".request.json") {
            continue;
        }

        let processing = shared_root
            .join("requests")
            .join("processing")
            .join(&file_name);
        if fs::rename(&path, &processing).is_err() {
            continue;
        }

        let outcome = handle_request(&processing, shared_root, indexes_root, embedding_override);
        let target_dir = if outcome.is_ok() { "done" } else { "failed" };
        let target = shared_root
            .join("requests")
            .join(target_dir)
            .join(&file_name);
        let _ = fs::rename(&processing, target);
        if let Err(err) = outcome {
            eprintln!("request failed: {err:#}");
        }
    }
    Ok(())
}

fn cleanup_old_files(
    shared_root: &Path,
    done_ttl_secs: u64,
    failed_ttl_secs: u64,
) -> anyhow::Result<()> {
    let done_ttl = Duration::from_secs(done_ttl_secs);
    cleanup_dir(&shared_root.join("requests").join("done"), done_ttl)?;
    cleanup_dir(
        &shared_root.join("requests").join("failed"),
        Duration::from_secs(failed_ttl_secs),
    )?;
    cleanup_response_tree(&shared_root.join("responses"), done_ttl)?;
    Ok(())
}

fn cleanup_dir(dir: &Path, ttl: Duration) -> anyhow::Result<()> {
    if ttl.is_zero() || !dir.exists() {
        return Ok(());
    }
    let now = std::time::SystemTime::now();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let Ok(metadata) = fs::metadata(&path) else {
            continue;
        };
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        let Ok(age) = now.duration_since(modified) else {
            continue;
        };
        if age >= ttl {
            let _ = fs::remove_file(path);
        }
    }
    Ok(())
}

fn cleanup_response_tree(dir: &Path, ttl: Duration) -> anyhow::Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    cleanup_tree_files(dir, ttl)?;
    cleanup_empty_child_dirs(dir)?;
    Ok(())
}

fn cleanup_tree_files(dir: &Path, ttl: Duration) -> anyhow::Result<()> {
    if ttl.is_zero() {
        return Ok(());
    }
    let now = std::time::SystemTime::now();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            cleanup_tree_files(&path, ttl)?;
            continue;
        }
        if !path.is_file() {
            continue;
        }
        let Ok(metadata) = fs::metadata(&path) else {
            continue;
        };
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        let Ok(age) = now.duration_since(modified) else {
            continue;
        };
        if age >= ttl {
            let _ = fs::remove_file(path);
        }
    }
    Ok(())
}

fn cleanup_empty_child_dirs(dir: &Path) -> anyhow::Result<bool> {
    if !dir.exists() {
        return Ok(true);
    }
    let mut empty = true;
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            if cleanup_empty_child_dirs(&path)? {
                let _ = fs::remove_dir(&path);
            } else {
                empty = false;
            }
        } else {
            empty = false;
        }
    }
    Ok(empty)
}

fn handle_request(
    path: &Path,
    shared_root: &Path,
    indexes_root: &Path,
    embedding_override: &EmbeddingConfigOverride,
) -> anyhow::Result<()> {
    let request: Request = read_json(path)?;
    match request {
        Request::Search(req) => {
            let out_path = response_path(shared_root, &req.client_id, &req.request_id);
            let response = match load_current_index_with_embedding_override(
                indexes_root,
                &req.dataset_id,
                embedding_override,
            )
            .and_then(|index| index.search(&req))
            {
                Ok(result) => Response::Ok(ResponseOk::Search(result)),
                Err(err) => Response::Error(ResponseError {
                    request_id: req.request_id,
                    message: err.to_string(),
                }),
            };
            atomic_write_json(&out_path, &response)?;
        }
        Request::DescribeDataset(req) => {
            let out_path = response_path(shared_root, &req.client_id, &req.request_id);
            let response = match load_current_index_with_embedding_override(
                indexes_root,
                &req.dataset_id,
                embedding_override,
            )
            .and_then(|index| index.describe(req.request_id.clone()))
            {
                Ok(result) => Response::Ok(ResponseOk::Dataset(result)),
                Err(err) => Response::Error(ResponseError {
                    request_id: req.request_id,
                    message: err.to_string(),
                }),
            };
            atomic_write_json(&out_path, &response)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleanup_response_tree_removes_files_past_ttl_and_empty_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let responses = dir.path().join("responses");
        let old_client = responses.join("old-client");
        let empty_client = responses.join("empty-client");
        fs::create_dir_all(&old_client).unwrap();
        fs::create_dir_all(&empty_client).unwrap();
        fs::write(old_client.join("old.response.json"), "{}").unwrap();

        cleanup_response_tree(&responses, Duration::from_secs(0)).unwrap();
        assert!(old_client.join("old.response.json").exists());
        assert!(!empty_client.exists());

        cleanup_response_tree(&responses, Duration::from_nanos(1)).unwrap();
        assert!(!old_client.exists());
    }
}
