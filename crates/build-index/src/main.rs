use std::path::PathBuf;

use clap::Parser;
use hybrid_shared_core::chunking::{ChunkMode, ChunkOptions};
use hybrid_shared_core::embedding::EmbeddingConfig;
use hybrid_shared_core::index::{build_index, BuildOptions};

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Build a searchable dataset index from schema.json and JSONL"
)]
struct Args {
    #[arg(long)]
    dataset: String,
    #[arg(long)]
    schema: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long, default_value = "indexes")]
    indexes_root: PathBuf,
    #[arg(long = "index-version")]
    index_version: Option<String>,
    #[arg(long)]
    embedding_model: Option<PathBuf>,
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    #[arg(long)]
    ort_dll: Option<PathBuf>,
    #[arg(long, default_value_t = 768)]
    embedding_dim: usize,
    #[arg(long, default_value_t = 8192)]
    max_input_tokens: usize,
    #[arg(long, default_value = "ruri-v3-onnx")]
    embedding_model_id: String,
    #[arg(long, default_value = "検索クエリ: ")]
    query_prefix: String,
    #[arg(long, default_value = "検索文書: ")]
    document_prefix: String,
    #[arg(long)]
    preload_model_to_memory: bool,
    #[arg(long, default_value = "none")]
    chunk_mode: String,
    #[arg(long, default_value_t = 1200)]
    chunk_size: usize,
    #[arg(long, default_value_t = 200)]
    chunk_overlap: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let embedding = match (&args.embedding_model, &args.tokenizer, &args.ort_dll) {
        (Some(model_path), Some(tokenizer_path), Some(runtime_library_path)) => {
            Some(EmbeddingConfig {
                model_path: model_path.clone(),
                tokenizer_path: tokenizer_path.clone(),
                runtime_library_path: runtime_library_path.clone(),
                dimension: args.embedding_dim,
                max_input_tokens: args.max_input_tokens,
                model_id: args.embedding_model_id.clone(),
                query_prefix: args.query_prefix.clone(),
                document_prefix: args.document_prefix.clone(),
                preload_model_to_memory: args.preload_model_to_memory,
            })
        }
        (None, None, None) => None,
        _ => {
            anyhow::bail!("--embedding-model, --tokenizer, and --ort-dll must be provided together")
        }
    };
    let dir = build_index(BuildOptions {
        dataset_id: args.dataset,
        schema_path: args.schema,
        input_path: args.input,
        indexes_root: args.indexes_root,
        version: args.index_version,
        embedding,
        chunking: ChunkOptions {
            mode: parse_chunk_mode(&args.chunk_mode)?,
            size: args.chunk_size,
            overlap: args.chunk_overlap,
        },
    })?;
    println!("index built: {}", dir.display());
    Ok(())
}

fn parse_chunk_mode(value: &str) -> anyhow::Result<ChunkMode> {
    match value {
        "none" => Ok(ChunkMode::None),
        "smart" => Ok(ChunkMode::Smart),
        _ => anyhow::bail!("--chunk-mode must be one of: none, smart"),
    }
}
