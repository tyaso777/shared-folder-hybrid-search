# Shared Folder Hybrid Search

Prototype search system for Windows environments where publishing an HTTP server is not allowed. The server and clients exchange JSON request/response files through a shared folder.

## Current Features

- Server-side index build from `schema.json` + flat JSONL.
- Dataset-specific JSON field mapping.
- Strict build-time validation for missing fields, duplicate primary keys, nested values, and filter type mismatch.
- Localhost browser UI on the user PC.
- Shared-folder RPC between client and server.
- Dynamic UI filters from dataset schema and facets.
- Tantivy + Lindera(IPADIC) BM25 text search.
- Optional ONNX Runtime + ruri-v3 style embedding.
- Optional HNSW vector search.
- RRF fusion when both BM25 and vector candidates are available.
- Filter-aware search: BM25 applies exact filters in Tantivy, and vector search uses one HNSW index with adaptive oversampling plus SQLite filter checks.

## Architecture

```text
Server operator
  build-index.exe
    schema.json + input.jsonl
      -> indexes/{dataset}/versions/{version}
      -> indexes/{dataset}/current.json

Search server
  search-server.exe
    watches shared/requests/pending
    reads current index
    writes shared/responses/{client_id}

User PC
  search-client.exe
    opens http://127.0.0.1:{port}
    browser UI -> local Rust client -> shared folder RPC
```

## Build

```powershell
cargo build
```

## Dependency Checks

GitHub Actions runs dependency checks on `main`, pull requests, manual dispatch, and every Monday.

- Rust vulnerabilities and licenses: `cargo-deny` with `deny.toml`.
- Python vulnerabilities: `pip-audit` against `scripts/onnx-export-requirements.txt`.
- Python licenses: `pip-licenses` against the ONNX export helper dependencies.

The Python dependency set is only for `scripts/export_ruri_v3_onnx.ps1`; the Rust search server/client do not require Python at runtime.

## Use Your Own JSON

For a real project, the server operator usually starts from two files:

- `schema.json`: tells the indexer which JSON fields are IDs, searchable text, display fields, source links, and filters.
- `input.jsonl`: one flat JSON object per line. Nested objects and arrays are rejected.

Example project layout:

```text
data/my_project/schema.json
data/my_project/input.jsonl
```

Example `input.jsonl`:

```jsonl
{"doc_id":"doc-001","title":"教育助言","body":"義務教育に関するリスクと対応方針。","source":"internal","source_uri":"https://example.local/doc-001","department":"教育","updated_at":"2026-05-01"}
{"doc_id":"doc-002","title":"契約確認","body":"契約更新時の確認事項とリスク評価。","source":"internal","source_uri":"https://example.local/doc-002","department":"法務","updated_at":"2026-05-10"}
```

Example `schema.json`:

```json
{
  "dataset_id": "my_project",
  "primary_key": "doc_id",
  "text_fields": ["title", "body"],
  "full_text_fields": ["title", "body"],
  "source_uri_field": "source_uri",
  "source_label_field": "source",
  "display_fields": ["title", "department", "updated_at", "source_uri"],
  "filter_fields": {
    "source": {
      "type": "keyword",
      "label": "Source",
      "ui": "select"
    },
    "department": {
      "type": "keyword",
      "label": "Department",
      "ui": "select"
    },
    "updated_at": {
      "type": "date",
      "label": "Updated At",
      "ui": "date_range"
    }
  }
}
```

Field meanings:

| Field | Meaning |
| --- | --- |
| `dataset_id` | Dataset name used by `build-index`, `search-server`, and `search-client`. |
| `primary_key` | Unique record ID field. Duplicate values fail the build. |
| `text_fields` | Fields joined into the searchable document text for BM25 and vector chunking. |
| `full_text_fields` | Dataset metadata for fields that represent the original text. The UI currently expands only the matched chunk, not the whole parent document. |
| `source_uri_field` | Field used by the UI's open/copy source actions. |
| `source_label_field` | Field used for source chips and source filters, such as `Wikibooks`, `Wikipedia`, or an internal system name. |
| `display_fields` | Fields copied into each result payload for result card display. |
| `filter_fields` | Fields exposed as UI filters and enforced by the search server. |

Build a BM25-only index:

```powershell
cargo run -p build-index -- `
  --dataset my_project `
  --schema data\my_project\schema.json `
  --input data\my_project\input.jsonl `
  --indexes-root indexes
```

Build a hybrid BM25 + vector index:

```powershell
cargo run -p build-index -- `
  --dataset my_project `
  --schema data\my_project\schema.json `
  --input data\my_project\input.jsonl `
  --indexes-root indexes `
  --embedding-model models\ruri-v3-onnx\model.onnx `
  --tokenizer models\ruri-v3-onnx\tokenizer.json `
  --ort-dll models\ruri-v3-onnx\onnxruntime.dll `
  --embedding-dim 768 `
  --max-input-tokens 512 `
  --chunk-mode smart `
  --chunk-size 1200 `
  --chunk-overlap 200
```

Start the shared-folder server:

```powershell
cargo run -p search-server -- `
  --shared-root shared_demo `
  --indexes-root indexes
```

Start the browser client in another terminal:

```powershell
cargo run -p search-client -- `
  --shared-root shared_demo `
  --dataset my_project
```

For company deployment, `--shared-root` can be a shared folder path accessible to both the server process and client PCs. Each client writes requests with its own `client_id`, so concurrent users do not mix responses.

## Demo: BM25 Only

Build a sample index:

```powershell
cargo run -p build-index -- `
  --dataset contracts `
  --schema examples/contracts/schema.json `
  --input examples/contracts/input.jsonl `
  --indexes-root indexes
```

## Demo: Japanese Wikibooks 50-Record Vector Sample

The recommended local demo dataset is `jawikibooks_vector_50`. It keeps setup time and vector build time manageable while still exercising BM25, Lindera, smart chunking, ONNX ruri-v3 embeddings, HNSW, RRF, filters, and the browser UI.

Download a Japanese Wikibooks pages/articles dump locally. Wikimedia-derived text is not committed to this repository.

Dump index:

```text
https://dumps.wikimedia.org/jawikibooks/latest/
```

Use a file such as:

```text
jawikibooks-latest-pages-articles.xml.bz2
```

For example, place the dump here:

```text
vendor/wikimedia/jawikibooks-latest-pages-articles.xml.bz2
```

Convert 50 pages into flat JSONL:

```powershell
cargo run -p convert-mediawiki-dump -- `
  --dataset jawikibooks_vector_50 `
  --input vendor\wikimedia\jawikibooks-latest-pages-articles.xml.bz2 `
  --output-dir examples\jawikibooks_vector_50 `
  --limit 50
```

Export ruri-v3 to ONNX if `models\ruri-v3-onnx` does not exist yet:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\export_ruri_v3_onnx.ps1
```

Then build the vector index:

```powershell
cargo run -p build-index -- `
  --dataset jawikibooks_vector_50 `
  --schema examples\jawikibooks_vector_50\schema.json `
  --input examples\jawikibooks_vector_50\input.jsonl `
  --indexes-root indexes `
  --embedding-model models\ruri-v3-onnx\model.onnx `
  --tokenizer models\ruri-v3-onnx\tokenizer.json `
  --ort-dll models\ruri-v3-onnx\onnxruntime.dll `
  --embedding-dim 768 `
  --max-input-tokens 512 `
  --chunk-mode smart `
  --chunk-size 1200 `
  --chunk-overlap 200
```

The converter stores page title, cleaned wiki text, URL, source, and conversion timestamp. The wiki-text cleaning is intentionally simple; it is suitable for search prototyping, not archival-quality rendering.

Wikimedia-derived text data is intentionally excluded from this repository. Users should download and convert it locally. Generated JSONL, SQLite DBs, Tantivy indexes, HNSW indexes, and any redistributed sample content remain subject to the source data license, such as CC BY-SA attribution/share-alike requirements. For larger local tests, change `--limit 50` to a larger value, but vector indexing on CPU can become slow.

The repository ignores generated local data paths such as:

```text
examples/jawiki_sample/
examples/jawiki_sample_test/
examples/jawikibooks_vector_50/
indexes/
models/
vendor/
```

Start the shared-folder server:

```powershell
cargo run -p search-server -- `
  --shared-root shared_demo `
  --indexes-root indexes `
  --done-ttl-secs 600 `
  --failed-ttl-secs 86400 `
  --cleanup-interval-secs 60
```

Start the local browser client in another terminal:

```powershell
cargo run -p search-client -- `
  --shared-root shared_demo `
  --dataset jawikibooks_vector_50
```

The browser client deletes response files after reading them. Use `--keep-responses` when debugging shared-folder traffic. The server deletes processed request files from `requests/done` after 10 minutes and `requests/failed` after 24 hours by default.

## Demo: BM25 + ONNX Vector

First export ruri-v3 to ONNX and copy `onnxruntime.dll` locally:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\export_ruri_v3_onnx.ps1
```

Then provide all embedding options together:

```powershell
cargo run -p build-index -- `
  --dataset contracts `
  --schema examples/contracts/schema.json `
  --input examples/contracts/input.jsonl `
  --indexes-root indexes `
  --embedding-model models\ruri-v3-onnx\model.onnx `
  --tokenizer models\ruri-v3-onnx\tokenizer.json `
  --ort-dll models\ruri-v3-onnx\onnxruntime.dll `
  --embedding-dim 768 `
  --max-input-tokens 512 `
  --chunk-mode smart `
  --chunk-size 1200 `
  --chunk-overlap 200
```

When embedding options are omitted, the system builds a BM25-only Tantivy index. When embedding options are provided, it also builds `hnsw/` and stores `embedding_config.json` in the index version. The search server automatically uses vector search and RRF fusion when the vector index is present.

Filters are applied before ranking as much as the current index layout allows. BM25 pushes `eq` and `in` filters into the Tantivy query. Vector search keeps a single HNSW index and increases the number of vector candidates when filters remove too many hits, then checks those candidates against SQLite filter values before RRF or result materialization. Range filters are still checked in SQLite, so they remain correct but are less selective before BM25 scoring.

For `cl-nagoya/ruri-v3-310m`, the builder and server use the official retrieval prefixes by default:

- Query embedding: `検索クエリ: {query}`
- Document embedding: `検索文書: {record searchable text}`

These defaults can be overridden with `--query-prefix` and `--document-prefix`, but existing vector indexes built without these prefixes should be rebuilt.

`--chunk-mode smart` keeps BM25 at document level but embeds vector chunks. The chunker prefers natural Japanese boundaries: blank lines and newlines first, then sentence punctuation such as `。！？`, then lighter punctuation such as `、`. If no suitable boundary is found, it falls back to a hard character boundary. Vector hits are aggregated back to the parent document by max similarity, and the browser UI shows the best vector chunk text for each result.

Chunk rows store only their own `chunk_text`, plus `chunk_id`, parent `record_id`, `chunk_index`, and character offsets. The larger parent document text is stored once in `records.searchable_text` and is not duplicated into every chunk row.

Search responses include `query_terms`, generated with the same Lindera tokenizer used by BM25. The browser UI highlights the full query and tokenized terms in snippets and vector-hit chunks. If `query_terms` is unavailable, the UI falls back to simple whitespace/punctuation splitting.

The browser UI supports `Hybrid`, `BM25 only`, and `Vector only` search modes. Result cards show score details, BM25/vector ranks when available, highlighted snippets, and a collapsible vector-hit chunk. The UI also includes filter reset and elapsed-time display for quick evaluation.

The UI also supports result granularity:

- `文書`: one card per parent record, with the best vector chunk shown inside the card.
- `チャンク`: one card per vector-hit chunk, so the same parent record may appear multiple times for different matching passages. If Vector is off, choosing chunk granularity automatically turns Vector on and shows a short notice.

## Dataset Schema

`schema.json` defines how flat JSONL records are interpreted.

```json
{
  "dataset_id": "contracts",
  "primary_key": "contract_id",
  "text_fields": ["title", "body", "notes"],
  "full_text_fields": ["title", "body", "notes"],
  "source_uri_field": "source_uri",
  "source_label_field": "source",
  "display_fields": ["title", "department", "updated_at", "source_uri"],
  "filter_fields": {
    "department": {
      "type": "keyword",
      "label": "Department",
      "ui": "select"
    },
    "updated_at": {
      "type": "date",
      "label": "Updated At",
      "ui": "date_range"
    }
  }
}
```

Input JSONL must be flat. Nested objects and arrays are rejected by design.

`source_uri_field` is used for `URLをコピー` and `原文を開く`; `source_label_field` is used for source chips and source badges. `full_text_fields` is retained as dataset metadata, but the current browser UI keeps expansion lightweight: `全文を見る` shows only the best vector-matched chunk text, not the full parent document.

## Shared Folder Protocol

Requests are written atomically to:

```text
shared/requests/pending/{request_id}.request.json
```

The server claims a request by renaming it to:

```text
shared/requests/processing/{request_id}.request.json
```

Responses are written to:

```text
shared/responses/{client_id}/{request_id}.response.json
```

This avoids mixing results between concurrent users.

## Current Limitations

- The ONNX embedder expects encoder output shaped `[batch, seq_len, hidden]` and uses masked mean pooling.
- HNSW is rebuilt during index construction. Incremental vector update/delete is intentionally not exposed yet.
- CPU embedding is slow for thousands of long pages. Use `--max-input-tokens 512`, `--chunk-mode smart`, or a smaller dataset for local vector-search tests.
- If no ONNX config is supplied, search is BM25-only.
- Strong security filters should be enforced server-side. The current design avoids filter-specific vector indexes; very selective vector filters may require larger oversampling or a future filtered ANN strategy.
- The shared folder should be protected with Windows ACLs because requests and results may contain sensitive text.
