# VersionWise

Version-aware technical documentation intelligence and comparison workspace.

## Project Structure

```text
D:\VersionWise
├── client
│   ├── package.json
│   ├── package-lock.json
│   ├── index.html
│   ├── vite.config.js
│   └── src
│       ├── main.jsx
│       ├── App.jsx
│       └── index.css
├── server
│   ├── requirements.txt
│   ├── app
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── schemas.py
│   │   └── ingestion.py
│   ├── scripts
│   │   └── ingest_docs.py
│   └── tests
│       ├── fixtures
│       │   ├── pydantic_v1_sample.md
│       │   └── pydantic_v2_sample.md
│       ├── test_health.py
│       ├── test_database.py
│       └── test_ingestion.py
├── .gitignore
├── .env.example
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.13.0 (using `.venv` virtual environment)
- Node.js 24.12.0 and npm 11.6.2

---

### Configuration

Environment variables can be configured via a local `.env` file (copied from `.env.example`):

- `APP_NAME`: Application title (default: `VersionWise`).
- `APP_ENV`: Environment mode (default: `development`).
- `MONGODB_URI`: MongoDB connection string (e.g. `mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority`).
- `MONGODB_DATABASE`: Target MongoDB database name (default: `versionwise`).

> **Note:** MongoDB connectivity is optional during initial development. If `MONGODB_URI` is unset, the backend starts cleanly without a database client. If `MONGODB_URI` is configured but unreachable at startup, the server logs a warning and starts in degraded mode.

---

### Backend Setup and Running

1. **Install dependencies** (into `.venv`):
   ```powershell
   & "D:\VersionWise\.venv\Scripts\python.exe" -m pip install -r server\requirements.txt
   ```

2. **Run tests**:
   ```powershell
   $env:PYTHONPATH="server"; & "D:\VersionWise\.venv\Scripts\python.exe" -m pytest server -v
   ```

3. **Start backend development server**:
   ```powershell
   & "D:\VersionWise\.venv\Scripts\python.exe" -m uvicorn app.main:app --app-dir server --reload
   ```

#### Health Endpoints
- `GET /health`: Core application liveness probe. Always returns HTTP 200 with `{"status": "ok"}`.
- `GET /health/db`: Database connectivity probe. Returns HTTP 200 with `{"status": "connected"}` if MongoDB is configured and reachable, or HTTP 503 with `{"status": "unavailable"}` if unconfigured or unreachable.

---

### Documentation Ingestion (Phase 3)

VersionWise includes a deterministic, heading-aware documentation ingestion pipeline designed for versioned technical docs (initially targeting Pydantic `1.10` and `2.12`).

#### Core Architecture
- **Heading-Aware Chunking (`server/app/ingestion.py`)**: Slices Markdown on heading boundaries (`#`..`######`), building hierarchical breadcrumbs (e.g. `Models > Custom Validators`), splitting oversized sections on paragraph boundaries, and generating deterministic SHA-256 `chunk_id`s and `content_hash`es.
- **Idempotent Storage & Stale Reconciliation**: Chunks are upserted into the `document_chunks` collection via PyMongo `bulk_write`. `created_at` is set strictly on insert via `$setOnInsert`; `updated_at` is updated only when content or relevant persisted metadata changes. Unchanged chunks are skipped. Stale chunks from previous versions of the exact page are cleanly purged only after 100% successful write. If any write or index error occurs, stale-chunk reconciliation is completely bypassed.
- **Strict Dry-Run Isolation**: Passing `--dry-run` performs full in-memory parsing, chunking, and Pydantic validation without instantiating a `MongoClient` or connecting to MongoDB.
- **Duplicate URL Pre-Validation**: In directory mode, the CLI builds and validates the complete file-to-canonical-URL mapping across all discovered files before reading, parsing, or connecting, immediately aborting on collision.
- **Directory Version Homogeneity**: A `--source-dir` invocation applies one explicit `--version` argument across all files in that directory. The directory path is checked for conflicting version indicators (e.g. specifying `--version 1.10` on a path containing `2.12`) to guard against misconfiguration. The caller remains responsible for ensuring all source documents in `--source-dir` belong to the specified version.

#### CLI Ingestion Command
To preview ingestion in memory without database writes (`--dry-run`):
```powershell
& "D:\VersionWise\.venv\Scripts\python.exe" server\scripts\ingest_docs.py --source-file "server\tests\fixtures\pydantic_v1_sample.md" --version "1.10" --url "https://docs.pydantic.dev/1.10/usage/validators/" --dry-run
```

To ingest into MongoDB Atlas:
```powershell
& "D:\VersionWise\.venv\Scripts\python.exe" server\scripts\ingest_docs.py --source-file "path\to\doc.md" --version "1.10" --url "https://docs.pydantic.dev/1.10/usage/models/"
```

Or ingest a version-homogeneous directory:
```powershell
& "D:\VersionWise\.venv\Scripts\python.exe" server\scripts\ingest_docs.py --source-dir "data/sources/pydantic/2.12/docs" --version "2.12" --url "https://docs.pydantic.dev/2.12/"
```

> **Corpus Notice:** Synthetic fixtures in `server/tests/fixtures/` are used solely for deterministic unit testing. The official Pydantic documentation corpus will be staged and verified in a subsequent phase.

---

### Embeddings & Vector Retrieval (Phase 4)

VersionWise incorporates a version-aware dense vector retrieval pipeline powered by dense semantic embeddings (configured for `sentence-transformers/all-MiniLM-L6-v2`, 384-dimensional dense vectors) and MongoDB Atlas Vector Search.

#### Core Retrieval Architecture
- **Pluggable Embedding Providers (`server/app/embeddings.py`)**:
  - `MockEmbeddingProvider`: Deterministic, zero-dependency provider for offline testing and in-memory dry-runs.
  - `SentenceTransformerEmbeddingProvider`: Lazily loaded model provider. Strictly enforces local-only loading (`local_files_only=True`); automatic model or tokenizer file downloads are prohibited and fail safely.
- **Sub-Window Overlapping Mean Pooling**: Chunks exceeding the verified tokenizer budget (254 content tokens) are partitioned into overlapping sub-windows (stride 127), encoded individually, combined via uniform mean pooling, and $L_2$-normalized. This ensures all tokens across the chunk contribute to the vector without silent truncation.
- **Strict Dry-Run Isolation**:
  - `--dry-run`: Text-only in-memory parsing and structural validation. Zero models loaded, zero embeddings generated, zero database connections.
  - `--dry-run-embeddings`: In-memory parsing and embedding generation with vector statistics. Strictly zero database connections, writes, or deletions.
- **Safe Ingestion & Backfill (`server/app/ingestion.py`)**:
  - Stored embeddings are fetched via standard MongoDB projection and validated locally in Python.
  - Unchanged chunks with valid 384-dimensional vectors are skipped.
  - Unchanged chunks with missing or invalid vectors are repaired/backfilled without modifying `created_at`.
  - Changed chunks are atomically updated with newly generated vectors. Retaining a stale vector on changed content is strictly prohibited.
  - Persistent mock vectors are completely prohibited: attempting live database writes with a mock provider unconditionally aborts.
- **Search Endpoint (`POST /api/v1/retrieval/search`)**:
  - Validates query (non-empty, non-whitespace, max 1,000 characters), library (`pydantic`), version (`1.10` or `2.12`), and `top_k` ($1 \le \text{top\_k} \le 50$).
  - Executes MongoDB Atlas `$vectorSearch` with cosine similarity, pre-filtering on both `library` and `version` to prevent cross-version contamination.
  - Returns full provenance: `canonical_url`, `page_title`, `heading_path`, `content`, and similarity `score`.
  - Sanitized error responses: Missing vector search indexes or database errors return HTTP 503 without leaking credentials or raw exception details.

#### MongoDB Atlas Vector Search Index Specification
Atlas Vector Search indexes must be created manually in the MongoDB Atlas UI or via the Atlas Admin API for the `document_chunks` collection:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "library"
    },
    {
      "type": "filter",
      "path": "version"
    }
  ]
}
```

> **Notice:** Live Atlas index creation, live Atlas semantic evaluation, and Gemini LLM integration are deferred to subsequent phases.

---

### Frontend Setup and Running

1. **Install dependencies**:
   ```powershell
   cd client
   npm ci
   ```

2. **Run development server**:
   ```powershell
   cd client
   npm run dev
   ```

3. **Build for production**:
   ```powershell
   cd client
   npm run build
   ```
