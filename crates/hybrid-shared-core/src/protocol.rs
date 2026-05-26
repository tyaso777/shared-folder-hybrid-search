use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::schema::DatasetSchema;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    Search(SearchRequest),
    DescribeDataset(DescribeDatasetRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub request_id: String,
    pub client_id: String,
    pub dataset_id: String,
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub filters: BTreeMap<String, FilterExpr>,
    #[serde(default)]
    pub search_mode: SearchMode,
    #[serde(default)]
    pub result_granularity: ResultGranularity,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    Hybrid,
    Bm25,
    Vector,
}

impl Default for SearchMode {
    fn default() -> Self {
        Self::Hybrid
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResultGranularity {
    Document,
    Chunk,
}

impl Default for ResultGranularity {
    fn default() -> Self {
        Self::Document
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescribeDatasetRequest {
    pub request_id: String,
    pub client_id: String,
    pub dataset_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterExpr {
    Eq {
        eq: Value,
    },
    In {
        r#in: Vec<Value>,
    },
    Range {
        gte: Option<Value>,
        lte: Option<Value>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Response {
    Ok(ResponseOk),
    Error(ResponseError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResponseOk {
    Search(SearchResponse),
    Dataset(DatasetDescription),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub request_id: String,
    pub dataset_id: String,
    pub index_version: String,
    #[serde(default)]
    pub query_terms: Vec<String>,
    pub results: Vec<SearchResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub result_id: String,
    pub record_id: String,
    pub score: f32,
    pub display: Value,
    pub snippet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_chunk: Option<MatchedChunk>,
    pub scores: ScoreBreakdown,
    pub payload: Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub final_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_score: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedChunk {
    pub chunk_id: String,
    pub chunk_index: usize,
    pub score: f32,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetDescription {
    pub request_id: String,
    pub dataset_id: String,
    pub index_version: String,
    pub schema: DatasetSchema,
    pub facets: BTreeMap<String, Vec<FacetValue>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetValue {
    pub value: Value,
    pub count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseError {
    pub request_id: String,
    pub message: String,
}

fn default_top_k() -> usize {
    20
}
