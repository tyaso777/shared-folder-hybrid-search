use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use axum::extract::{Path, State};
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use hybrid_shared_core::config::{default_config_path, SharedSearchConfig};
use hybrid_shared_core::protocol::{
    DescribeDatasetRequest, FilterExpr, Request, Response, ResultGranularity, SearchMode,
    SearchRequest,
};
use hybrid_shared_core::shared_folder::{
    atomic_write_json, ensure_layout, read_json, request_path, response_path,
};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Local browser UI client for shared-folder search"
)]
struct Args {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    shared_root: Option<PathBuf>,
    #[arg(long)]
    dataset: Option<String>,
    #[arg(long)]
    no_open: Option<bool>,
    #[arg(long)]
    keep_responses: Option<bool>,
}

#[derive(Debug)]
struct ResolvedArgs {
    shared_root: PathBuf,
    dataset: String,
    no_open: bool,
    keep_responses: bool,
    default_top_k: usize,
    request_timeout_secs: u64,
    browser_shutdown_secs: u64,
    search_poll_interval_ms: u64,
    client_port: u16,
}

#[derive(Clone)]
struct AppState {
    shared_root: PathBuf,
    client_id: String,
    dataset_id: String,
    last_seen: std::sync::Arc<Mutex<Instant>>,
    keep_responses: bool,
    default_top_k: usize,
    request_timeout: Duration,
    browser_shutdown: Duration,
    search_poll_interval_ms: u64,
}

#[derive(Debug, Deserialize)]
struct UiSearchRequest {
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    filters: BTreeMap<String, FilterExpr>,
    #[serde(default)]
    search_mode: SearchMode,
    #[serde(default)]
    result_granularity: ResultGranularity,
}

#[derive(Debug, Serialize)]
struct SubmitResponse {
    request_id: String,
}

#[derive(Debug, Serialize)]
struct ClientConfigResponse {
    default_top_k: usize,
    request_timeout_secs: u64,
    search_poll_interval_ms: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = resolve_args(Args::parse())?;
    ensure_layout(&args.shared_root)?;
    let state = AppState {
        shared_root: args.shared_root,
        client_id: format!("client-{}", Uuid::new_v4()),
        dataset_id: args.dataset,
        last_seen: std::sync::Arc::new(Mutex::new(Instant::now())),
        keep_responses: args.keep_responses,
        default_top_k: args.default_top_k,
        request_timeout: Duration::from_secs(args.request_timeout_secs),
        browser_shutdown: Duration::from_secs(args.browser_shutdown_secs),
        search_poll_interval_ms: args.search_poll_interval_ms,
    };

    let app = Router::new()
        .route("/", get(index_html))
        .route("/api/heartbeat", post(heartbeat))
        .route("/api/client-config", get(client_config))
        .route("/api/dataset", get(describe_dataset))
        .route("/api/search", post(submit_search))
        .route("/api/jobs/:request_id", get(get_job))
        .with_state(state.clone());

    let listener = TcpListener::bind(("127.0.0.1", args.client_port)).await?;
    let addr = listener.local_addr()?;
    let url = format!("http://{addr}/");
    println!("client UI: {url}");
    if !args.no_open {
        let _ = open::that(&url);
    }

    tokio::spawn(shutdown_when_browser_disappears(
        state.last_seen.clone(),
        state.browser_shutdown,
    ));
    axum::serve(listener, app).await?;
    Ok(())
}

fn resolve_args(args: Args) -> anyhow::Result<ResolvedArgs> {
    let config_path = args.config.or_else(default_config_path);
    let config = match config_path {
        Some(path) if path.exists() => SharedSearchConfig::load_resolved(&path)?,
        _ => SharedSearchConfig::default(),
    };
    let dataset = args
        .dataset
        .or_else(|| config.dataset_id())
        .ok_or_else(|| {
            anyhow::anyhow!("dataset is required; set --dataset or shared-search.toml dataset_id")
        })?;
    let shared_root = args
        .shared_root
        .or(config.shared_root)
        .unwrap_or_else(|| PathBuf::from("shared_demo"));
    Ok(ResolvedArgs {
        shared_root,
        dataset,
        no_open: args.no_open.or(config.no_open).unwrap_or(false),
        keep_responses: args
            .keep_responses
            .or(config.keep_responses)
            .unwrap_or(false),
        default_top_k: config.default_top_k.unwrap_or(20),
        request_timeout_secs: config.request_timeout_secs.unwrap_or(60),
        browser_shutdown_secs: config.browser_shutdown_secs.unwrap_or(30),
        search_poll_interval_ms: config.search_poll_interval_ms.unwrap_or(300),
        client_port: config.client_port.unwrap_or(0),
    })
}

async fn index_html() -> Html<&'static str> {
    Html(include_str!("ui.html"))
}

async fn heartbeat(State(state): State<AppState>) -> impl IntoResponse {
    *state.last_seen.lock().await = Instant::now();
    Json(serde_json::json!({"ok": true}))
}

async fn client_config(State(state): State<AppState>) -> impl IntoResponse {
    Json(ClientConfigResponse {
        default_top_k: state.default_top_k,
        request_timeout_secs: state.request_timeout.as_secs(),
        search_poll_interval_ms: state.search_poll_interval_ms,
    })
}

async fn describe_dataset(State(state): State<AppState>) -> impl IntoResponse {
    match roundtrip(
        &state,
        Request::DescribeDataset(DescribeDatasetRequest {
            request_id: Uuid::new_v4().to_string(),
            client_id: state.client_id.clone(),
            dataset_id: state.dataset_id.clone(),
        }),
        state.request_timeout,
    )
    .await
    {
        Ok(response) => Json(response),
        Err(err) => Json(Response::Error(
            hybrid_shared_core::protocol::ResponseError {
                request_id: String::new(),
                message: err.to_string(),
            },
        )),
    }
}

async fn submit_search(
    State(state): State<AppState>,
    Json(input): Json<UiSearchRequest>,
) -> impl IntoResponse {
    let request_id = Uuid::new_v4().to_string();
    let request = Request::Search(SearchRequest {
        request_id: request_id.clone(),
        client_id: state.client_id.clone(),
        dataset_id: state.dataset_id.clone(),
        query: input.query,
        top_k: input.top_k,
        filters: input.filters,
        search_mode: input.search_mode,
        result_granularity: input.result_granularity,
    });
    let path = request_path(&state.shared_root, &request_id);
    match atomic_write_json(&path, &request) {
        Ok(()) => Json(serde_json::to_value(SubmitResponse { request_id }).unwrap()),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

async fn get_job(
    State(state): State<AppState>,
    Path(request_id): Path<String>,
) -> impl IntoResponse {
    let path = response_path(&state.shared_root, &state.client_id, &request_id);
    if !path.exists() {
        return Json(serde_json::json!({ "status": "pending" }));
    }
    match read_json::<Response>(&path) {
        Ok(response) => {
            if !state.keep_responses {
                let _ = std::fs::remove_file(&path);
            }
            Json(serde_json::to_value(response).unwrap())
        }
        Err(err) => Json(serde_json::json!({ "status": "error", "message": err.to_string() })),
    }
}

async fn roundtrip(
    state: &AppState,
    request: Request,
    timeout: Duration,
) -> anyhow::Result<Response> {
    let (request_id, client_id) = match &request {
        Request::Search(r) => (r.request_id.clone(), r.client_id.clone()),
        Request::DescribeDataset(r) => (r.request_id.clone(), r.client_id.clone()),
    };
    atomic_write_json(&request_path(&state.shared_root, &request_id), &request)?;
    let response = response_path(&state.shared_root, &client_id, &request_id);
    let started = Instant::now();
    loop {
        if response.exists() {
            let value = read_json(&response)?;
            if !state.keep_responses {
                let _ = std::fs::remove_file(&response);
            }
            return Ok(value);
        }
        if started.elapsed() > timeout {
            anyhow::bail!("timeout waiting for server response");
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

async fn shutdown_when_browser_disappears(
    last_seen: std::sync::Arc<Mutex<Instant>>,
    browser_shutdown: Duration,
) {
    loop {
        tokio::time::sleep(Duration::from_secs(10)).await;
        if last_seen.lock().await.elapsed() > browser_shutdown {
            eprintln!("browser heartbeat expired; exiting");
            std::process::exit(0);
        }
    }
}

fn default_top_k() -> usize {
    20
}
