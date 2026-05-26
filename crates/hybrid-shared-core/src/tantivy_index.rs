use std::collections::BTreeMap;
use std::path::Path;

use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, QueryParser, TermQuery};
use tantivy::schema::{
    Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value as _, STORED, STRING,
};
use tantivy::tokenizer::TokenStream;
use tantivy::{doc, Index, Term};

use serde_json::Value;

use crate::protocol::FilterExpr;
use crate::schema::{DatasetSchema, PreparedRecord};

#[derive(Debug, Clone)]
pub struct TextHit {
    pub record_id: String,
    pub score: f32,
}

pub struct TantivyTextIndex {
    index: Index,
    reader: tantivy::IndexReader,
    f_record_id: Field,
    f_text: Field,
    filter_fields: BTreeMap<String, Field>,
}

impl TantivyTextIndex {
    pub fn create(
        path: &Path,
        dataset_schema: &DatasetSchema,
        records: &[PreparedRecord],
    ) -> anyhow::Result<Self> {
        if path.exists() {
            std::fs::remove_dir_all(path)?;
        }
        std::fs::create_dir_all(path)?;
        let (schema, f_record_id, f_text, filter_fields) = build_schema(dataset_schema);
        let index = Index::create_in_dir(path, schema)?;
        register_lindera(&index);
        let mut writer = index.writer(50_000_000)?;
        for record in records {
            let mut document = doc!(
                f_record_id => record.record_id.clone(),
                f_text => record.searchable_text.clone(),
            );
            for (field_name, value) in &record.filters {
                let Some(field) = filter_fields.get(field_name) else {
                    continue;
                };
                if let Some(text) = value_to_string(value) {
                    document.add_text(*field, text);
                }
            }
            writer.add_document(document)?;
        }
        writer.commit()?;
        let reader = index.reader()?;
        Ok(Self {
            index,
            reader,
            f_record_id,
            f_text,
            filter_fields,
        })
    }

    pub fn open(path: &Path, dataset_schema: &DatasetSchema) -> anyhow::Result<Self> {
        let index = Index::open_in_dir(path)?;
        register_lindera(&index);
        let schema = index.schema();
        let f_record_id = schema.get_field("record_id")?;
        let f_text = schema.get_field("text")?;
        let filter_fields = dataset_schema
            .filter_fields
            .keys()
            .filter_map(|field_name| {
                schema
                    .get_field(&filter_field_name(field_name))
                    .ok()
                    .map(|field| (field_name.clone(), field))
            })
            .collect();
        let reader = index.reader()?;
        Ok(Self {
            index,
            reader,
            f_record_id,
            f_text,
            filter_fields,
        })
    }

    pub fn search(
        &self,
        query: &str,
        filters: &BTreeMap<String, FilterExpr>,
        limit: usize,
    ) -> anyhow::Result<Vec<TextHit>> {
        if query.trim().is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let parsed = self.filtered_query(query, filters);
        let searcher = self.reader.searcher();
        let hits = searcher.search(&parsed, &TopDocs::with_limit(limit))?;
        let mut out = Vec::with_capacity(hits.len());
        for (score, addr) in hits {
            let doc = searcher.doc::<tantivy::schema::document::TantivyDocument>(addr)?;
            if let Some(value) = doc.get_first(self.f_record_id) {
                if let Some(record_id) = value.as_str() {
                    out.push(TextHit {
                        record_id: record_id.to_string(),
                        score,
                    });
                }
            }
        }
        Ok(out)
    }

    pub fn query_terms(&self, query: &str) -> Vec<String> {
        self.tokenized_terms(query)
    }

    fn tokenized_or_query(&self, query: &str) -> Box<dyn tantivy::query::Query> {
        let tokens = self.tokenized_terms(query);
        if tokens.is_empty() {
            let parser = QueryParser::for_index(&self.index, vec![self.f_text]);
            return parser
                .parse_query(query)
                .unwrap_or_else(|_| Box::new(BooleanQuery::new(Vec::new())));
        }
        let clauses = tokens
            .into_iter()
            .map(|token| {
                let term = Term::from_field_text(self.f_text, &token);
                (
                    Occur::Should,
                    Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs))
                        as Box<dyn tantivy::query::Query>,
                )
            })
            .collect::<Vec<_>>();
        Box::new(BooleanQuery::from(clauses))
    }

    fn tokenized_terms(&self, query: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        if let Ok(mut analyzer) = self.index.tokenizer_for_field(self.f_text) {
            let mut stream = analyzer.token_stream(query);
            while stream.advance() {
                let token = stream.token();
                if !token.text.trim().is_empty() {
                    tokens.push(token.text.clone());
                }
            }
        }
        tokens
    }

    fn filtered_query(
        &self,
        query: &str,
        filters: &BTreeMap<String, FilterExpr>,
    ) -> Box<dyn tantivy::query::Query> {
        let text_query = self.tokenized_or_query(query);
        let filter_queries = self.filter_queries(filters);
        if filter_queries.is_empty() {
            return text_query;
        }
        let mut clauses = Vec::with_capacity(filter_queries.len() + 1);
        clauses.push((Occur::Must, text_query));
        clauses.extend(filter_queries.into_iter().map(|query| (Occur::Must, query)));
        Box::new(BooleanQuery::from(clauses))
    }

    fn filter_queries(
        &self,
        filters: &BTreeMap<String, FilterExpr>,
    ) -> Vec<Box<dyn tantivy::query::Query>> {
        let mut queries = Vec::new();
        for (field_name, expr) in filters {
            let Some(field) = self.filter_fields.get(field_name).copied() else {
                continue;
            };
            match expr {
                FilterExpr::Eq { eq } => {
                    if let Some(text) = value_to_string(eq) {
                        queries.push(term_query(field, &text));
                    }
                }
                FilterExpr::In { r#in } => {
                    let terms = r#in
                        .iter()
                        .filter_map(value_to_string)
                        .map(|text| (Occur::Should, term_query(field, &text)))
                        .collect::<Vec<_>>();
                    if !terms.is_empty() {
                        queries.push(Box::new(BooleanQuery::from(terms)));
                    }
                }
                FilterExpr::Range { .. } => {
                    // Numeric/date ranges remain enforced by SQLite. Tantivy pre-filters
                    // exact-match filters without creating separate indexes per filter.
                }
            }
        }
        queries
    }
}

fn build_schema(dataset_schema: &DatasetSchema) -> (Schema, Field, Field, BTreeMap<String, Field>) {
    let mut builder = Schema::builder();
    let text_indexing = TextFieldIndexing::default()
        .set_tokenizer("ja")
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let text_options = TextOptions::default()
        .set_indexing_options(text_indexing)
        .set_stored();
    let record_id = builder.add_text_field("record_id", STRING | STORED);
    let text = builder.add_text_field("text", text_options);
    let mut filter_fields = BTreeMap::new();
    for field_name in dataset_schema.filter_fields.keys() {
        let field = builder.add_text_field(&filter_field_name(field_name), STRING | STORED);
        filter_fields.insert(field_name.clone(), field);
    }
    (builder.build(), record_id, text, filter_fields)
}

fn term_query(field: Field, text: &str) -> Box<dyn tantivy::query::Query> {
    Box::new(TermQuery::new(
        Term::from_field_text(field, text),
        IndexRecordOption::Basic,
    ))
}

fn filter_field_name(field: &str) -> String {
    let mut out = String::from("filter_");
    for ch in field.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    out
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

fn register_lindera(index: &Index) {
    use lindera::dictionary::load_dictionary;
    use lindera::mode::Mode;
    use lindera::segmenter::Segmenter;
    use lindera_tantivy::tokenizer::LinderaTokenizer;

    let dictionary = load_dictionary("embedded://ipadic")
        .expect("failed to load embedded Lindera IPADIC dictionary");
    let segmenter = Segmenter::new(Mode::Normal, dictionary, None);
    index
        .tokenizers()
        .register("ja", LinderaTokenizer::from_segmenter(segmenter));
}
