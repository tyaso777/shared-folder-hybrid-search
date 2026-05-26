use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::chunking::{build_chunks, ChunkOptions, PreparedChunk};
use crate::embedding::{resolve_config_paths, EmbeddingConfig, OnnxEmbedder};
use crate::protocol::{
    DatasetDescription, FacetValue, FilterExpr, MatchedChunk, ResultGranularity, ScoreBreakdown,
    SearchMode, SearchRequest, SearchResponse, SearchResult,
};
use crate::schema::{DatasetSchema, PreparedRecord};
use crate::tantivy_index::TantivyTextIndex;
use crate::vector_index::{HnswVectorIndex, VectorHit};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildMeta {
    pub dataset_id: String,
    pub index_version: String,
    pub built_at: String,
    pub record_count: usize,
    pub chunk_count: usize,
    pub engine: String,
}

#[derive(Debug, Clone)]
pub struct BuildOptions {
    pub dataset_id: String,
    pub schema_path: PathBuf,
    pub input_path: PathBuf,
    pub indexes_root: PathBuf,
    pub version: Option<String>,
    pub embedding: Option<EmbeddingConfig>,
    pub chunking: ChunkOptions,
}

#[derive(Debug, Clone)]
pub struct SearchIndex {
    pub dataset_id: String,
    pub index_version: String,
    pub root: PathBuf,
    pub schema: DatasetSchema,
    pub embedding: Option<EmbeddingConfig>,
}

struct CandidateSearch {
    ranked: Vec<(String, f32)>,
    vector_chunks: BTreeMap<String, MatchedChunk>,
    vector_chunk_hits: Vec<ChunkCandidate>,
    bm25_ranks: BTreeMap<String, usize>,
    vector_ranks: BTreeMap<String, usize>,
}

struct ChunkCandidate {
    record_id: String,
    rank: usize,
    chunk: MatchedChunk,
}

struct ChunkRow {
    record_id: String,
    chunk_index: usize,
    start_char: usize,
    end_char: usize,
    text: String,
}

pub fn build_index(options: BuildOptions) -> anyhow::Result<PathBuf> {
    let schema_text = fs::read_to_string(&options.schema_path)
        .with_context(|| format!("read schema {}", options.schema_path.display()))?;
    let schema: DatasetSchema = serde_json::from_str(&schema_text)?;
    if schema.dataset_id != options.dataset_id {
        return Err(anyhow!(
            "dataset id mismatch: CLI `{}` vs schema `{}`",
            options.dataset_id,
            schema.dataset_id
        ));
    }

    let values = read_jsonl(&options.input_path)?;
    let records = schema.prepare_records(values)?;
    let chunks = build_chunks(&records, &options.chunking)?;
    let version = options
        .version
        .unwrap_or_else(|| Utc::now().format("%Y%m%d-%H%M%S").to_string());
    let version_dir = options
        .indexes_root
        .join(&options.dataset_id)
        .join("versions")
        .join(&version);
    fs::create_dir_all(&version_dir)?;

    fs::copy(&options.schema_path, version_dir.join("schema.json"))?;
    write_sqlite_index(&version_dir.join("records.db"), &schema, &records, &chunks)?;
    write_facets(&version_dir.join("filter_facets.json"), &records)?;
    TantivyTextIndex::create(&version_dir.join("tantivy"), &schema, &records)?;

    if let Some(mut embedding_cfg) = options.embedding.clone() {
        resolve_config_paths(&mut embedding_cfg, Path::new("."));
        fs::write(
            version_dir.join("embedding_config.json"),
            serde_json::to_vec_pretty(&embedding_cfg)?,
        )?;
        build_vector_index(&version_dir, &embedding_cfg, &chunks)?;
    }

    let meta = BuildMeta {
        dataset_id: options.dataset_id.clone(),
        index_version: version.clone(),
        built_at: Utc::now().to_rfc3339(),
        record_count: records.len(),
        chunk_count: chunks.len(),
        engine: if options.embedding.is_some() {
            "tantivy-lindera-bm25+hnsw-onnx-rrf".to_string()
        } else {
            "tantivy-lindera-bm25".to_string()
        },
    };
    fs::write(
        version_dir.join("build_meta.json"),
        serde_json::to_vec_pretty(&meta)?,
    )?;

    let current_path = options
        .indexes_root
        .join(&options.dataset_id)
        .join("current.json");
    if let Some(parent) = current_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        current_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "dataset_id": options.dataset_id,
            "index_version": version,
            "path": version_dir
        }))?,
    )?;

    Ok(version_dir)
}

pub fn load_current_index(indexes_root: &Path, dataset_id: &str) -> anyhow::Result<SearchIndex> {
    let current_path = indexes_root.join(dataset_id).join("current.json");
    let current: Value = serde_json::from_slice(&fs::read(&current_path)?)?;
    let root = current
        .get("path")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("current.json missing path"))?;
    let index_version = current
        .get("index_version")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let schema: DatasetSchema = serde_json::from_slice(&fs::read(root.join("schema.json"))?)?;
    let embedding = {
        let path = root.join("embedding_config.json");
        if path.exists() {
            Some(serde_json::from_slice(&fs::read(path)?)?)
        } else {
            None
        }
    };
    Ok(SearchIndex {
        dataset_id: dataset_id.to_string(),
        index_version,
        root,
        schema,
        embedding,
    })
}

impl SearchIndex {
    pub fn search(&self, req: &SearchRequest) -> anyhow::Result<SearchResponse> {
        let conn = Connection::open(self.root.join("records.db"))?;
        let query_terms = self.query_terms(&req.query)?;
        let candidates = self.search_candidates(&conn, req)?;
        let mut results = if req.result_granularity == ResultGranularity::Chunk {
            materialize_chunk_results(&conn, req, &candidates)?
        } else {
            materialize_ranked_ids(&conn, &candidates.ranked, req, &candidates)?
        };
        if results.is_empty() && !req.query.trim().is_empty() {
            results = search_like(&conn, req)?;
        }

        Ok(SearchResponse {
            request_id: req.request_id.clone(),
            dataset_id: self.dataset_id.clone(),
            index_version: self.index_version.clone(),
            query_terms,
            results,
        })
    }

    fn query_terms(&self, query: &str) -> anyhow::Result<Vec<String>> {
        let tantivy_dir = self.root.join("tantivy");
        let tokens = if tantivy_dir.exists() {
            TantivyTextIndex::open(&tantivy_dir, &self.schema)?.query_terms(query)
        } else {
            Vec::new()
        };
        Ok(normalize_query_terms(query, tokens))
    }

    fn search_candidates(
        &self,
        conn: &Connection,
        req: &SearchRequest,
    ) -> anyhow::Result<CandidateSearch> {
        let fetch_limit = req.top_k.max(1).saturating_mul(20).max(100);
        let mut ranked_lists: Vec<Vec<String>> = Vec::new();
        let mut vector_chunks = BTreeMap::new();
        let mut vector_chunk_hits = Vec::new();
        let mut bm25_ranks = BTreeMap::new();
        let mut vector_ranks = BTreeMap::new();

        let tantivy_dir = self.root.join("tantivy");
        if req.search_mode != SearchMode::Vector && tantivy_dir.exists() {
            let text_index = TantivyTextIndex::open(&tantivy_dir, &self.schema)?;
            let hits = text_index.search(&req.query, &req.filters, fetch_limit)?;
            let ranking = hits
                .into_iter()
                .map(|hit| hit.record_id)
                .collect::<Vec<_>>();
            bm25_ranks = rank_map(&ranking);
            ranked_lists.push(ranking);
        }

        if req.search_mode != SearchMode::Bm25 {
            if let Some(embedding_cfg) = &self.embedding {
                let hnsw_dir = self.root.join("hnsw");
                if hnsw_dir.join("map.tsv").exists() {
                    let embedder = OnnxEmbedder::new(embedding_cfg.clone())?;
                    let qvec = embedder.embed_query(&req.query)?;
                    let vector_index = HnswVectorIndex::load(&hnsw_dir, embedder.dimension())?;
                    let (ranking, chunks, chunk_hits) =
                        filtered_vector_candidates(conn, &vector_index, &qvec, req, fetch_limit)?;
                    vector_ranks = rank_map(&ranking);
                    ranked_lists.push(ranking);
                    vector_chunks = chunks;
                    vector_chunk_hits = chunk_hits;
                }
            }
        }

        let ranked = if ranked_lists.len() == 1 {
            ranked_lists
                .pop()
                .unwrap_or_default()
                .into_iter()
                .take(fetch_limit)
                .enumerate()
                .map(|(rank, id)| (id, 1.0 / ((rank + 1) as f32)))
                .collect()
        } else {
            rrf_fuse(ranked_lists, 60, fetch_limit)
        };

        Ok(CandidateSearch {
            ranked,
            vector_chunks,
            vector_chunk_hits,
            bm25_ranks,
            vector_ranks,
        })
    }

    pub fn describe(&self, request_id: String) -> anyhow::Result<DatasetDescription> {
        let facets_path = self.root.join("filter_facets.json");
        let facets = if facets_path.exists() {
            serde_json::from_slice(&fs::read(facets_path)?)?
        } else {
            BTreeMap::new()
        };
        Ok(DatasetDescription {
            request_id,
            dataset_id: self.dataset_id.clone(),
            index_version: self.index_version.clone(),
            schema: self.schema.clone(),
            facets,
        })
    }
}

fn filtered_vector_candidates(
    conn: &Connection,
    vector_index: &HnswVectorIndex,
    query_vector: &[f32],
    req: &SearchRequest,
    fetch_limit: usize,
) -> anyhow::Result<(
    Vec<String>,
    BTreeMap<String, MatchedChunk>,
    Vec<ChunkCandidate>,
)> {
    const MAX_VECTOR_FETCH: usize = 5_000;

    let target = fetch_limit.max(req.top_k.max(1));
    let mut vector_fetch = fetch_limit.max(req.top_k.max(1));
    let max_fetch = MAX_VECTOR_FETCH.max(vector_fetch);

    loop {
        let hits = vector_index.search(query_vector, vector_fetch);
        let candidates = aggregate_vector_hits(conn, hits, &req.filters)?;
        let enough = if req.result_granularity == ResultGranularity::Chunk {
            candidates.2.len() >= target
        } else {
            candidates.0.len() >= target
        };
        if enough || vector_fetch >= max_fetch {
            return Ok(candidates);
        }
        let next = vector_fetch.saturating_mul(3).min(max_fetch);
        if next == vector_fetch {
            return Ok(candidates);
        }
        vector_fetch = next;
    }
}

fn search_like(conn: &Connection, req: &SearchRequest) -> anyhow::Result<Vec<SearchResult>> {
    let mut sql = String::from(
        "SELECT r.record_id, r.display_json, r.payload_json, r.searchable_text, 0.0 AS rank \
         FROM records r WHERE 1=1",
    );
    let mut args: Vec<rusqlite::types::Value> = Vec::new();
    for term in req
        .query
        .split_whitespace()
        .filter(|s| !s.trim().is_empty())
    {
        sql.push_str(" AND r.searchable_text LIKE ?");
        args.push(format!("%{term}%").into());
    }
    append_filter_sql(&mut sql, &mut args, &req.filters);
    sql.push_str(" ORDER BY r.record_id LIMIT ?");
    args.push((req.top_k.max(1) as i64).into());
    let mut stmt = conn.prepare(&sql)?;
    materialize_rows(&mut stmt, args, &req.query)
}

fn materialize_ranked_ids(
    conn: &Connection,
    ranked: &[(String, f32)],
    req: &SearchRequest,
    candidates: &CandidateSearch,
) -> anyhow::Result<Vec<SearchResult>> {
    if ranked.is_empty() {
        return Ok(Vec::new());
    }
    let score_map = ranked.iter().cloned().collect::<BTreeMap<_, _>>();
    let mut out = Vec::new();
    for chunk in ranked.chunks(200) {
        let mut sql = String::from(
            "SELECT r.record_id, r.display_json, r.payload_json, r.searchable_text, 0.0 AS rank FROM records r WHERE r.record_id IN (",
        );
        let mut args: Vec<rusqlite::types::Value> = Vec::new();
        for (i, (record_id, _)) in chunk.iter().enumerate() {
            if i > 0 {
                sql.push(',');
            }
            sql.push('?');
            args.push(record_id.clone().into());
        }
        sql.push(')');
        append_filter_sql(&mut sql, &mut args, &req.filters);
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = materialize_rows(&mut stmt, args, &req.query)?;
        for row in &mut rows {
            row.score = *score_map.get(&row.record_id).unwrap_or(&0.0);
            row.matched_chunk = candidates.vector_chunks.get(&row.record_id).cloned();
            row.scores = ScoreBreakdown {
                final_score: row.score,
                bm25_rank: candidates.bm25_ranks.get(&row.record_id).copied(),
                vector_rank: candidates.vector_ranks.get(&row.record_id).copied(),
                vector_score: row.matched_chunk.as_ref().map(|chunk| chunk.score),
            };
            if let Some(chunk) = &row.matched_chunk {
                row.snippet = make_snippet(&chunk.text, &req.query);
            }
        }
        out.extend(rows);
    }
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(req.top_k.max(1));
    Ok(out)
}

fn materialize_chunk_results(
    conn: &Connection,
    req: &SearchRequest,
    candidates: &CandidateSearch,
) -> anyhow::Result<Vec<SearchResult>> {
    let mut out = Vec::new();
    for candidate in &candidates.vector_chunk_hits {
        if out.len() >= req.top_k.max(1) {
            break;
        }
        let mut sql = String::from(
            "SELECT r.record_id, r.display_json, r.payload_json, r.searchable_text, 0.0 AS rank \
             FROM records r WHERE r.record_id = ?",
        );
        let mut args: Vec<rusqlite::types::Value> = vec![candidate.record_id.clone().into()];
        append_filter_sql(&mut sql, &mut args, &req.filters);
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = materialize_rows(&mut stmt, args, &req.query)?;
        let Some(mut row) = rows.pop() else {
            continue;
        };
        row.result_id = candidate.chunk.chunk_id.clone();
        row.score = candidate.chunk.score;
        row.snippet = make_snippet(&candidate.chunk.text, &req.query);
        row.matched_chunk = Some(candidate.chunk.clone());
        row.scores = ScoreBreakdown {
            final_score: row.score,
            bm25_rank: candidates.bm25_ranks.get(&row.record_id).copied(),
            vector_rank: Some(candidate.rank),
            vector_score: Some(candidate.chunk.score),
        };
        out.push(row);
    }
    Ok(out)
}

fn materialize_rows(
    stmt: &mut rusqlite::Statement<'_>,
    args: Vec<rusqlite::types::Value>,
    query: &str,
) -> anyhow::Result<Vec<SearchResult>> {
    let rows = stmt.query_map(rusqlite::params_from_iter(args), |row| {
        let record_id: String = row.get(0)?;
        let display_json: String = row.get(1)?;
        let payload_json: String = row.get(2)?;
        let searchable_text: String = row.get(3)?;
        let rank: f64 = row.get(4)?;
        Ok((record_id, display_json, payload_json, searchable_text, rank))
    })?;

    let mut results = Vec::new();
    for row in rows {
        let (record_id, display_json, payload_json, searchable_text, rank) = row?;
        results.push(SearchResult {
            result_id: record_id.clone(),
            record_id,
            score: (-rank) as f32,
            display: serde_json::from_str(&display_json)?,
            snippet: make_snippet(&searchable_text, query),
            matched_chunk: None,
            scores: ScoreBreakdown {
                final_score: (-rank) as f32,
                ..ScoreBreakdown::default()
            },
            payload: serde_json::from_str(&payload_json)?,
        });
    }
    Ok(results)
}

fn read_jsonl(path: &Path) -> anyhow::Result<Vec<Value>> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)
            .with_context(|| format!("parse JSONL line {}", index + 1))?;
        values.push(value);
    }
    Ok(values)
}

fn write_sqlite_index(
    path: &Path,
    schema: &DatasetSchema,
    records: &[PreparedRecord],
    chunks: &[PreparedChunk],
) -> anyhow::Result<()> {
    if path.exists() {
        fs::remove_file(path)?;
    }
    let mut conn = Connection::open(path)?;
    conn.execute_batch(
        r#"
        PRAGMA journal_mode = WAL;
        CREATE TABLE records (
            dataset_id TEXT NOT NULL,
            record_id TEXT NOT NULL,
            searchable_text TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            display_json TEXT NOT NULL,
            indexed_at TEXT NOT NULL,
            row_hash TEXT NOT NULL,
            PRIMARY KEY(dataset_id, record_id)
        );
        CREATE TABLE filter_values (
            record_id TEXT NOT NULL,
            field_name TEXT NOT NULL,
            string_value TEXT,
            number_value REAL,
            bool_value INTEGER
        );
        CREATE INDEX idx_filter_values ON filter_values(field_name, string_value, number_value, bool_value);
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            record_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            start_char INTEGER NOT NULL,
            end_char INTEGER NOT NULL,
            chunk_text TEXT NOT NULL
        );
        CREATE INDEX idx_chunks_record_id ON chunks(record_id);
        "#,
    )?;
    let tx = conn.transaction()?;
    let now = Utc::now().to_rfc3339();
    for record in records {
        tx.execute(
            "INSERT INTO records VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                schema.dataset_id,
                record.record_id,
                record.searchable_text,
                record.payload_json,
                record.display_json,
                now,
                record.row_hash,
            ],
        )?;
        for (field, value) in &record.filters {
            let string_value = value_to_string(value);
            let number_value = value.as_f64();
            let bool_value = value.as_bool().map(i64::from);
            tx.execute(
                "INSERT INTO filter_values(record_id, field_name, string_value, number_value, bool_value) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![record.record_id, field, string_value, number_value, bool_value],
            )?;
        }
    }
    for chunk in chunks {
        tx.execute(
            "INSERT INTO chunks(chunk_id, record_id, chunk_index, start_char, end_char, chunk_text) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                chunk.chunk_id,
                chunk.record_id,
                chunk.chunk_index as i64,
                chunk.start_char as i64,
                chunk.end_char as i64,
                chunk.text
            ],
        )?;
    }
    tx.commit()?;
    Ok(())
}

fn build_vector_index(
    version_dir: &Path,
    embedding_cfg: &EmbeddingConfig,
    chunks: &[PreparedChunk],
) -> anyhow::Result<()> {
    let embedder = OnnxEmbedder::new(embedding_cfg.clone())?;
    let mut items = Vec::with_capacity(chunks.len());
    let total = chunks.len();
    let mut done = 0usize;
    for batch in chunks.chunks(4) {
        let texts = batch
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect::<Vec<_>>();
        let vectors = embedder.embed_documents(&texts)?;
        for (chunk, vector) in batch.iter().zip(vectors) {
            items.push((chunk.chunk_id.clone(), vector));
        }
        done += batch.len();
        if done % 20 == 0 || done == total {
            eprintln!("embedded {done}/{total} chunks");
        }
    }
    let index = HnswVectorIndex::build(embedder.dimension(), items)?;
    index.save(&version_dir.join("hnsw"))?;
    Ok(())
}

fn aggregate_vector_hits(
    conn: &Connection,
    hits: Vec<VectorHit>,
    filters: &BTreeMap<String, FilterExpr>,
) -> anyhow::Result<(
    Vec<String>,
    BTreeMap<String, MatchedChunk>,
    Vec<ChunkCandidate>,
)> {
    if !table_exists(conn, "chunks")? {
        let mut filtered = Vec::new();
        for hit in hits {
            if record_matches_filters(conn, &hit.record_id, filters)? {
                filtered.push(hit.record_id);
            }
        }
        let ranking = unique_record_ids(filtered.into_iter());
        return Ok((ranking, BTreeMap::new(), Vec::new()));
    }

    let mut stmt = if column_exists(conn, "chunks", "chunk_text")? {
        conn.prepare(
            "SELECT record_id, chunk_index, 0 AS start_char, 0 AS end_char, chunk_text \
             FROM chunks WHERE chunk_id = ?",
        )?
    } else {
        conn.prepare(
            "SELECT c.record_id, c.chunk_index, c.start_char, c.end_char, r.searchable_text \
             FROM chunks c JOIN records r ON r.record_id = c.record_id WHERE c.chunk_id = ?",
        )?
    };
    let mut ranking = Vec::new();
    let mut chunks = BTreeMap::new();
    let mut chunk_hits = Vec::new();
    for hit in hits {
        let row = stmt.query_row(params![hit.record_id], |row| {
            Ok(ChunkRow {
                record_id: row.get(0)?,
                chunk_index: row.get::<_, i64>(1)? as usize,
                start_char: row.get::<_, i64>(2)? as usize,
                end_char: row.get::<_, i64>(3)? as usize,
                text: row.get(4)?,
            })
        });
        let Ok(row) = row else {
            continue;
        };
        if !record_matches_filters(conn, &row.record_id, filters)? {
            continue;
        }
        let matched_chunk = MatchedChunk {
            chunk_id: hit.record_id,
            chunk_index: row.chunk_index,
            score: 1.0 - hit.distance,
            text: if row.start_char == 0 && row.end_char == 0 {
                row.text
            } else {
                slice_chars_owned(&row.text, row.start_char, row.end_char)
            },
        };
        chunk_hits.push(ChunkCandidate {
            record_id: row.record_id.clone(),
            rank: chunk_hits.len() + 1,
            chunk: matched_chunk.clone(),
        });
        if chunks.contains_key(&row.record_id) {
            continue;
        }
        ranking.push(row.record_id.clone());
        chunks.insert(row.record_id, matched_chunk);
    }
    Ok((ranking, chunks, chunk_hits))
}

fn record_matches_filters(
    conn: &Connection,
    record_id: &str,
    filters: &BTreeMap<String, FilterExpr>,
) -> anyhow::Result<bool> {
    if filters.is_empty() {
        return Ok(true);
    }
    let mut sql = String::from("SELECT 1 FROM records r WHERE r.record_id = ?");
    let mut args: Vec<rusqlite::types::Value> = vec![record_id.to_string().into()];
    append_filter_sql(&mut sql, &mut args, filters);
    sql.push_str(" LIMIT 1");
    let found = conn
        .query_row(&sql, rusqlite::params_from_iter(args), |row| {
            row.get::<_, i64>(0)
        })
        .optional()?;
    Ok(found.is_some())
}

fn unique_record_ids(ids: impl Iterator<Item = String>) -> Vec<String> {
    let mut seen = BTreeMap::new();
    let mut out = Vec::new();
    for id in ids {
        if seen.insert(id.clone(), ()).is_none() {
            out.push(id);
        }
    }
    out
}

fn rank_map(ranking: &[String]) -> BTreeMap<String, usize> {
    ranking
        .iter()
        .enumerate()
        .map(|(index, id)| (id.clone(), index + 1))
        .collect()
}

fn table_exists(conn: &Connection, table: &str) -> anyhow::Result<bool> {
    let exists: i64 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?",
        params![table],
        |row| row.get(0),
    )?;
    Ok(exists > 0)
}

fn column_exists(conn: &Connection, table: &str, column: &str) -> anyhow::Result<bool> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
    for row in rows {
        if row? == column {
            return Ok(true);
        }
    }
    Ok(false)
}

fn slice_chars_owned(text: &str, start: usize, end: usize) -> String {
    if start >= end {
        return String::new();
    }
    let start_byte = char_to_byte(text, start);
    let end_byte = char_to_byte(text, end);
    text[start_byte..end_byte].to_string()
}

fn char_to_byte(text: &str, char_pos: usize) -> usize {
    if char_pos == 0 {
        return 0;
    }
    text.char_indices()
        .nth(char_pos)
        .map(|(byte, _)| byte)
        .unwrap_or(text.len())
}

fn normalize_query_terms(query: &str, tokenizer_terms: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut terms = Vec::new();
    for term in std::iter::once(query.trim().to_string())
        .chain(query.split_whitespace().map(str::to_string))
        .chain(tokenizer_terms)
    {
        let term = term.trim().to_string();
        if !is_highlight_term(&term) {
            continue;
        }
        let key = term.to_lowercase();
        if seen.insert(key) {
            terms.push(term);
        }
    }
    terms.sort_by_key(|term| std::cmp::Reverse(term.chars().count()));
    terms.truncate(32);
    terms
}

fn is_highlight_term(term: &str) -> bool {
    let mut chars = term.chars();
    let count = chars.clone().count();
    if count < 2 {
        return false;
    }
    chars.any(|ch| ch.is_alphanumeric() || !ch.is_ascii())
}

fn rrf_fuse(lists: Vec<Vec<String>>, k: usize, limit: usize) -> Vec<(String, f32)> {
    let mut scores: BTreeMap<String, f32> = BTreeMap::new();
    for list in lists {
        for (rank, id) in list.into_iter().enumerate() {
            *scores.entry(id).or_insert(0.0) += 1.0 / ((k + rank + 1) as f32);
        }
    }
    let mut items = scores.into_iter().collect::<Vec<_>>();
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    items.truncate(limit);
    items
}

fn write_facets(path: &Path, records: &[PreparedRecord]) -> anyhow::Result<()> {
    let mut counts: BTreeMap<String, BTreeMap<String, (Value, u64)>> = BTreeMap::new();
    for record in records {
        for (field, value) in &record.filters {
            let key = value_to_string(value).unwrap_or_default();
            let entry = counts.entry(field.clone()).or_default();
            let item = entry.entry(key).or_insert_with(|| (value.clone(), 0));
            item.1 += 1;
        }
    }
    let facets: BTreeMap<String, Vec<FacetValue>> = counts
        .into_iter()
        .map(|(field, values)| {
            let mut items: Vec<_> = values
                .into_values()
                .map(|(value, count)| FacetValue { value, count })
                .collect();
            items.sort_by(|a, b| b.count.cmp(&a.count));
            (field, items)
        })
        .collect();
    fs::write(path, serde_json::to_vec_pretty(&facets)?)?;
    Ok(())
}

fn append_filter_sql(
    sql: &mut String,
    args: &mut Vec<rusqlite::types::Value>,
    filters: &BTreeMap<String, FilterExpr>,
) {
    for (field, expr) in filters {
        match expr {
            FilterExpr::Eq { eq } => {
                sql.push_str(" AND EXISTS (SELECT 1 FROM filter_values fv WHERE fv.record_id = r.record_id AND fv.field_name = ? AND fv.string_value = ?)");
                args.push(field.clone().into());
                args.push(value_to_string(eq).unwrap_or_default().into());
            }
            FilterExpr::In { r#in } => {
                if r#in.is_empty() {
                    continue;
                }
                sql.push_str(" AND EXISTS (SELECT 1 FROM filter_values fv WHERE fv.record_id = r.record_id AND fv.field_name = ? AND fv.string_value IN (");
                args.push(field.clone().into());
                for (i, value) in r#in.iter().enumerate() {
                    if i > 0 {
                        sql.push(',');
                    }
                    sql.push('?');
                    args.push(value_to_string(value).unwrap_or_default().into());
                }
                sql.push_str("))");
            }
            FilterExpr::Range { gte, lte } => {
                sql.push_str(" AND EXISTS (SELECT 1 FROM filter_values fv WHERE fv.record_id = r.record_id AND fv.field_name = ?");
                args.push(field.clone().into());
                if let Some(value) = gte {
                    if let Some(n) = value.as_f64() {
                        sql.push_str(" AND fv.number_value >= ?");
                        args.push(n.into());
                    } else {
                        sql.push_str(" AND fv.string_value >= ?");
                        args.push(value_to_string(value).unwrap_or_default().into());
                    }
                }
                if let Some(value) = lte {
                    if let Some(n) = value.as_f64() {
                        sql.push_str(" AND fv.number_value <= ?");
                        args.push(n.into());
                    } else {
                        sql.push_str(" AND fv.string_value <= ?");
                        args.push(value_to_string(value).unwrap_or_default().into());
                    }
                }
                sql.push(')');
            }
        }
    }
}

fn make_snippet(text: &str, query: &str) -> String {
    let max_chars = 180;
    if query.trim().is_empty() {
        return text.chars().take(max_chars).collect();
    }
    let lower = text.to_lowercase();
    let needle = query
        .split_whitespace()
        .next()
        .unwrap_or(query)
        .to_lowercase();
    let start_byte = lower.find(&needle).unwrap_or(0);
    let start = text[..start_byte].chars().count().saturating_sub(40);
    text.chars().skip(start).take(max_chars).collect()
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Null => Some(String::new()),
        Value::Array(_) | Value::Object(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;
    use tempfile::tempdir;

    use super::*;
    use crate::protocol::FilterExpr;

    #[test]
    fn builds_and_searches_with_filters() {
        let dir = tempdir().unwrap();
        let schema_path = dir.path().join("schema.json");
        let input_path = dir.path().join("input.jsonl");
        fs::write(
            &schema_path,
            serde_json::to_vec(&json!({
                "dataset_id": "contracts",
                "primary_key": "contract_id",
                "text_fields": ["title", "body"],
                "display_fields": ["title", "department"],
                "filter_fields": {
                    "department": { "type": "keyword", "label": "部署", "ui": "select" }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            &input_path,
            concat!(
                "{\"contract_id\":\"c-001\",\"title\":\"契約更新\",\"body\":\"更新時のリスク確認\",\"department\":\"法務\"}\n",
                "{\"contract_id\":\"c-002\",\"title\":\"経費精算\",\"body\":\"領収書を確認\",\"department\":\"経理\"}\n"
            ),
        )
        .unwrap();

        build_index(BuildOptions {
            dataset_id: "contracts".to_string(),
            schema_path,
            input_path,
            indexes_root: dir.path().join("indexes"),
            version: Some("v1".to_string()),
            embedding: None,
            chunking: ChunkOptions::default(),
        })
        .unwrap();

        let index = load_current_index(&dir.path().join("indexes"), "contracts").unwrap();
        let mut filters = BTreeMap::new();
        filters.insert(
            "department".to_string(),
            FilterExpr::Eq {
                eq: json!("法務")
            },
        );
        let response = index
            .search(&SearchRequest {
                request_id: "r1".to_string(),
                client_id: "c1".to_string(),
                dataset_id: "contracts".to_string(),
                query: "契約更新".to_string(),
                top_k: 10,
                filters,
                search_mode: SearchMode::Hybrid,
                result_granularity: ResultGranularity::Document,
            })
            .unwrap();

        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].record_id, "c-001");
    }

    #[test]
    fn falls_back_to_substring_search_for_japanese_terms() {
        let dir = tempdir().unwrap();
        let schema_path = dir.path().join("schema.json");
        let input_path = dir.path().join("input.jsonl");
        fs::write(
            &schema_path,
            serde_json::to_vec(&json!({
                "dataset_id": "contracts",
                "primary_key": "contract_id",
                "text_fields": ["title", "body"],
                "display_fields": ["title"],
                "filter_fields": {}
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            &input_path,
            "{\"contract_id\":\"c-001\",\"title\":\"稟議申請\",\"body\":\"契約書ドラフトとリスク評価を添付する\"}\n",
        )
        .unwrap();

        build_index(BuildOptions {
            dataset_id: "contracts".to_string(),
            schema_path,
            input_path,
            indexes_root: dir.path().join("indexes"),
            version: Some("v1".to_string()),
            embedding: None,
            chunking: ChunkOptions::default(),
        })
        .unwrap();

        let index = load_current_index(&dir.path().join("indexes"), "contracts").unwrap();
        let response = index
            .search(&SearchRequest {
                request_id: "r1".to_string(),
                client_id: "c1".to_string(),
                dataset_id: "contracts".to_string(),
                query: "リスク".to_string(),
                top_k: 10,
                filters: BTreeMap::new(),
                search_mode: SearchMode::Hybrid,
                result_granularity: ResultGranularity::Document,
            })
            .unwrap();

        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].record_id, "c-001");
    }
}
