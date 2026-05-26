use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use hybrid_shared_core::index::load_current_index;
use hybrid_shared_core::protocol::{Request, Response, ResponseError, ResponseOk};
use hybrid_shared_core::shared_folder::{
    atomic_write_json, ensure_layout, read_json, response_path,
};

#[derive(Debug, Parser)]
#[command(author, version, about = "Shared-folder search server")]
struct Args {
    #[arg(long, default_value = "shared_demo")]
    shared_root: PathBuf,
    #[arg(long, default_value = "indexes")]
    indexes_root: PathBuf,
    #[arg(long, default_value_t = 2)]
    poll_seconds: u64,
    #[arg(long, default_value_t = 600)]
    done_ttl_secs: u64,
    #[arg(long, default_value_t = 86_400)]
    failed_ttl_secs: u64,
    #[arg(long, default_value_t = 60)]
    cleanup_interval_secs: u64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    ensure_layout(&args.shared_root)?;
    cleanup_old_files(&args.shared_root, args.done_ttl_secs, args.failed_ttl_secs)?;
    let mut last_cleanup = Instant::now();
    println!("watching {}", args.shared_root.display());
    loop {
        process_once(&args.shared_root, &args.indexes_root)?;
        if last_cleanup.elapsed() >= Duration::from_secs(args.cleanup_interval_secs) {
            cleanup_old_files(&args.shared_root, args.done_ttl_secs, args.failed_ttl_secs)?;
            last_cleanup = Instant::now();
        }
        thread::sleep(Duration::from_secs(args.poll_seconds));
    }
}

fn process_once(shared_root: &Path, indexes_root: &Path) -> anyhow::Result<()> {
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

        let outcome = handle_request(&processing, shared_root, indexes_root);
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
    cleanup_dir(
        &shared_root.join("requests").join("done"),
        Duration::from_secs(done_ttl_secs),
    )?;
    cleanup_dir(
        &shared_root.join("requests").join("failed"),
        Duration::from_secs(failed_ttl_secs),
    )?;
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

fn handle_request(path: &Path, shared_root: &Path, indexes_root: &Path) -> anyhow::Result<()> {
    let request: Request = read_json(path)?;
    match request {
        Request::Search(req) => {
            let out_path = response_path(shared_root, &req.client_id, &req.request_id);
            let response = match load_current_index(indexes_root, &req.dataset_id)
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
            let response = match load_current_index(indexes_root, &req.dataset_id)
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
