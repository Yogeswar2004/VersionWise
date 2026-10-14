import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pymongo.errors import BulkWriteError, OperationFailure, PyMongoError

from app.config import Settings
from app.embeddings import (
    EmbeddingDependencyMissingError,
    EmbeddingModelConfigurationError,
    EmbeddingModelUnavailableError,
    MockEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    get_embedding_provider,
    validate_numeric_vector,
)
from app.ingestion import store_and_reconcile_chunks
from app.main import app as main_app
from app.retrieval import (
    VectorSearchUnavailableError,
    build_vector_search_pipeline,
    perform_vector_search,
)
from app.schemas import DocumentChunk, RetrievalQuery, RetrievalResponse, RetrievalResult
from scripts.ingest_docs import main as cli_main

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# 1. Vector Dimension Validation
def test_validate_numeric_vector_dimensions() -> None:
    valid_vec = [1.0 / math.sqrt(384)] * 384
    res = validate_numeric_vector(valid_vec, expected_dim=384, check_normalized=True)
    assert len(res) == 384

    # Wrong dimension 383
    with pytest.raises(ValueError, match="dimension mismatch"):
        validate_numeric_vector([1.0] * 383, expected_dim=384, check_normalized=False)

    # Wrong dimension 385
    with pytest.raises(ValueError, match="dimension mismatch"):
        validate_numeric_vector([1.0] * 385, expected_dim=384, check_normalized=False)


# 2. NaN, Infinity, Booleans, Non-numeric, and Zero-norm Validation
def test_validate_numeric_vector_rejections() -> None:
    # Booleans are rejected (not treated as 0 or 1)
    bool_vec = [True] * 384
    with pytest.raises(ValueError, match="must be a float or int, not boolean"):
        validate_numeric_vector(bool_vec, expected_dim=384, check_normalized=False)

    # Non-numeric string
    str_vec = ["0.5"] * 384
    with pytest.raises(ValueError, match="must be a float or int"):
        validate_numeric_vector(str_vec, expected_dim=384, check_normalized=False)

    # NaN
    nan_vec = [float("nan")] * 384
    with pytest.raises(ValueError, match="must be a finite number"):
        validate_numeric_vector(nan_vec, expected_dim=384, check_normalized=False)

    # Infinity
    inf_vec = [float("inf")] * 384
    with pytest.raises(ValueError, match="must be a finite number"):
        validate_numeric_vector(inf_vec, expected_dim=384, check_normalized=False)

    # Zero norm (cannot be normalized)
    zero_vec = [0.0] * 384
    with pytest.raises(ValueError, match="zero or near-zero norm"):
        validate_numeric_vector(zero_vec, expected_dim=384, check_normalized=True)

    # Not normalized when check_normalized=True
    unnorm_vec = [2.0] * 384
    with pytest.raises(ValueError, match="not unit normalized"):
        validate_numeric_vector(unnorm_vec, expected_dim=384, check_normalized=True)


# 3. Deterministic Mock Provider Behavior
def test_mock_embedding_shape_and_determinism() -> None:
    provider = MockEmbeddingProvider(dim=384)
    v1 = provider.embed_text("pydantic custom validators")
    v2 = provider.embed_text("pydantic custom validators")
    v3 = provider.embed_text("different text content")

    assert len(v1) == 384
    assert v1 == v2
    assert v1 != v3
    norm = math.sqrt(sum(x * x for x in v1))
    assert abs(norm - 1.0) < 1e-4

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        provider.embed_text("   \n\t ")


# 4. Oversized Text Windowing and Pooling
def test_subwindow_mean_pooling_oversized_text() -> None:
    # A. Test MockEmbeddingProvider long text pooling
    provider = MockEmbeddingProvider(dim=384)
    long_text = "Paragraph content for testing. " * 50  # > 1000 chars
    vec = provider.embed_text(long_text)
    assert len(vec) == 384
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-4
    assert all(math.isfinite(x) for x in vec)

    # B. Test SentenceTransformerEmbeddingProvider sub-windowing with mocked model and tokenizer
    mock_st_provider = SentenceTransformerEmbeddingProvider("sentence-transformers/all-MiniLM-L6-v2", expected_dim=384)
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.max_seq_length = 256
    mock_model.tokenizer = mock_tokenizer
    mock_tokenizer.build_inputs_with_special_tokens.return_value = [101, 102]  # 2 special tokens

    # Simulate tokenizing long sequence: 600 tokens (> 254 capacity)
    fake_token_ids = list(range(600))
    mock_tokenizer.encode.return_value = fake_token_ids
    mock_tokenizer.decode.side_effect = lambda tokens, **kwargs: f"subtext with {len(tokens)} tokens"

    # Encode returns a valid unit vector for each sub-window
    unit_subvec = [1.0 / math.sqrt(384)] * 384
    mock_model.encode.return_value = MagicMock(tolist=lambda: unit_subvec)

    mock_st_provider._model = mock_model
    mock_st_provider._tokenizer = mock_tokenizer
    mock_st_provider._max_seq_length = 256
    mock_st_provider._window_capacity = 254

    final_vec = mock_st_provider.embed_text(long_text)
    assert len(final_vec) == 384
    final_norm = math.sqrt(sum(x * x for x in final_vec))
    assert abs(final_norm - 1.0) < 1e-4
    assert all(math.isfinite(x) for x in final_vec)
    # Verify model.encode was called multiple times for sub-windows
    assert mock_model.encode.call_count >= 2


# 5. Schema Validation for DocumentChunk and RetrievalQuery
def test_retrieval_query_and_chunk_schema_validation() -> None:
    valid_vec = [1.0 / math.sqrt(384)] * 384
    chunk = DocumentChunk(
        chunk_id="chk1",
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/",
        page_title="Title",
        heading_path="Title",
        chunk_index=0,
        content="Sample content",
        content_hash="h1",
        char_count=14,
        embedding=valid_vec,
    )
    assert len(chunk.embedding) == 384

    # Chunk with invalid embedding dimension
    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="chk2",
            library="pydantic",
            version="1.10",
            canonical_url="https://docs.pydantic.dev/1.10/",
            page_title="Title",
            heading_path="Title",
            chunk_index=0,
            content="Sample content",
            content_hash="h2",
            char_count=14,
            embedding=[1.0] * 100,
        )

    # Retrieval query validation
    q = RetrievalQuery(query="  how to validate fields  ", version="1.10")
    assert q.query == "how to validate fields"
    assert q.version == "1.10"
    assert q.top_k == 5

    # Empty query raises ValidationError
    with pytest.raises(ValidationError):
        RetrievalQuery(query="   ", version="1.10")

    # Unsupported version raises ValidationError
    with pytest.raises(ValidationError):
        RetrievalQuery(query="query", version="3.0")

    # Unsupported library raises ValidationError
    with pytest.raises(ValidationError):
        RetrievalQuery(query="query", version="1.10", library="fastapi")

    # Out of bounds top_k raises ValidationError
    with pytest.raises(ValidationError):
        RetrievalQuery(query="query", version="1.10", top_k=0)
    with pytest.raises(ValidationError):
        RetrievalQuery(query="query", version="1.10", top_k=51)


# 6. Unchanged Valid Chunks Skip Embedding Work
def test_unchanged_chunk_with_valid_vector_skipped() -> None:
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col

    valid_vec = [1.0 / math.sqrt(384)] * 384
    mock_col.find.return_value = [
        {
            "chunk_id": "c1",
            "content_hash": "hash1",
            "heading_path": "Path",
            "page_title": "Title",
            "source_release": "v1.10.19",
            "embedding": valid_vec,
        }
    ]

    chunk = DocumentChunk(
        chunk_id="c1",
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
        page_title="Title",
        heading_path="Path",
        chunk_index=0,
        content="Content",
        content_hash="hash1",
        char_count=7,
        source_release="v1.10.19",
        embedding=valid_vec,
    )

    summary = store_and_reconcile_chunks(
        db=mock_db,
        chunks=[chunk],
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
    )

    assert summary.success is True
    assert summary.chunks_skipped == 1
    assert summary.chunks_updated == 0
    assert not mock_col.bulk_write.called


# 7. Missing or Invalid Vectors are Backfilled
def test_unchanged_chunk_with_missing_or_invalid_vector_backfilled() -> None:
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col

    # Existing chunk matches metadata, but embedding is None (missing)
    mock_col.find.return_value = [
        {
            "chunk_id": "c1",
            "content_hash": "hash1",
            "heading_path": "Path",
            "page_title": "Title",
            "source_release": "v1.10.19",
            "embedding": None,
        }
    ]

    bulk_res = MagicMock()
    bulk_res.upserted_count = 0
    bulk_res.modified_count = 1
    mock_col.bulk_write.return_value = bulk_res

    valid_vec = [1.0 / math.sqrt(384)] * 384
    chunk = DocumentChunk(
        chunk_id="c1",
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
        page_title="Title",
        heading_path="Path",
        chunk_index=0,
        content="Content",
        content_hash="hash1",
        char_count=7,
        source_release="v1.10.19",
        embedding=valid_vec,
    )

    summary = store_and_reconcile_chunks(
        db=mock_db,
        chunks=[chunk],
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
    )

    assert summary.success is True
    assert summary.chunks_updated == 1
    assert summary.chunks_skipped == 0
    assert mock_col.bulk_write.called
    op = mock_col.bulk_write.call_args[0][0][0]
    # Backfill update: only embedding and updated_at are set; created_at is untouched
    assert "$set" in op._doc
    assert op._doc["$set"]["embedding"] == valid_vec
    assert "updated_at" in op._doc["$set"]
    assert "$setOnInsert" not in op._doc


# 8. Changed Metadata Triggers Re-Embedding / Re-Write
def test_changed_metadata_triggers_reembedding() -> None:
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col

    valid_vec = [1.0 / math.sqrt(384)] * 384
    mock_col.find.return_value = [
        {
            "chunk_id": "c1",
            "content_hash": "hash1",
            "heading_path": "Old Path",
            "page_title": "Title",
            "source_release": "v1.10.19",
            "embedding": valid_vec,
        }
    ]

    bulk_res = MagicMock()
    bulk_res.upserted_count = 0
    bulk_res.modified_count = 1
    mock_col.bulk_write.return_value = bulk_res

    new_vec = [1.0 / math.sqrt(384)] * 384
    chunk = DocumentChunk(
        chunk_id="c1",
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
        page_title="Title",
        heading_path="New Path",  # Metadata changed
        chunk_index=0,
        content="Content",
        content_hash="hash1",
        char_count=7,
        source_release="v1.10.19",
        embedding=new_vec,
    )

    summary = store_and_reconcile_chunks(
        db=mock_db,
        chunks=[chunk],
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
    )

    assert summary.chunks_updated == 1
    op = mock_col.bulk_write.call_args[0][0][0]
    assert op._doc["$set"]["heading_path"] == "New Path"
    assert op._doc["$set"]["embedding"] == new_vec


# 9. Changed Content Without New Vector Aborts
def test_changed_content_without_vector_aborts() -> None:
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col

    mock_col.find.return_value = [
        {
            "chunk_id": "c1",
            "content_hash": "old_hash",
            "heading_path": "Path",
            "page_title": "Title",
            "source_release": "v1.10.19",
            "embedding": [1.0 / math.sqrt(384)] * 384,
        }
    ]

    # Incoming chunk has new content_hash, but embedding is None
    chunk = DocumentChunk(
        chunk_id="c1",
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
        page_title="Title",
        heading_path="Path",
        chunk_index=0,
        content="New Content",
        content_hash="new_hash",
        char_count=11,
        source_release="v1.10.19",
        embedding=None,  # Missing vector!
    )

    summary = store_and_reconcile_chunks(
        db=mock_db,
        chunks=[chunk],
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
    )

    assert summary.success is False
    assert "Changed chunk lacks newly generated embedding vector" in summary.error_message
    assert not mock_col.bulk_write.called
    assert not mock_col.delete_many.called


# 10. Embedding Failure Prevents All Writes and Deletions
def test_embedding_failure_aborts_before_db_writes() -> None:
    sample_file = FIXTURES_DIR / "pydantic_v1_sample.md"

    with patch("app.embeddings.SentenceTransformerEmbeddingProvider.embed_chunk") as mock_embed:
        mock_embed.side_effect = RuntimeError("Simulated model out of memory error")
        with patch("app.database.create_mongo_client") as mock_client:
            code = cli_main(
                [
                    "--source-file",
                    str(sample_file),
                    "--version",
                    "1.10",
                    "--url",
                    "https://docs.pydantic.dev/1.10/usage/validators/",
                ]
            )
            assert code == 1
            # Verified: create_mongo_client was never called because embedding failed first
            assert not mock_client.called


# 11. Bulk-Write Error Bypasses Stale Reconciliation
def test_bulk_write_error_bypasses_stale_reconciliation() -> None:
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col
    mock_col.find.return_value = [{"chunk_id": "c2", "content_hash": "h2"}]
    mock_col.bulk_write.side_effect = BulkWriteError({"nUpserted": 0, "nModified": 0, "writeErrors": []})

    valid_vec = [1.0 / math.sqrt(384)] * 384
    chunk = DocumentChunk(
        chunk_id="c1",
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
        page_title="Title",
        heading_path="Path",
        chunk_index=0,
        content="Content",
        content_hash="h1",
        char_count=7,
        embedding=valid_vec,
    )

    summary = store_and_reconcile_chunks(
        db=mock_db,
        chunks=[chunk],
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
    )

    assert summary.success is False
    assert summary.partial_write is True
    assert not mock_col.delete_many.called


# 12. Live Persistence Rejects the Mock Provider
def test_cli_ingest_prohibits_mock_vectors_on_live_db() -> None:
    sample_file = FIXTURES_DIR / "pydantic_v1_sample.md"
    with patch("app.database.create_mongo_client") as mock_client:
        code = cli_main(
            [
                "--source-file",
                str(sample_file),
                "--version",
                "1.10",
                "--url",
                "https://docs.pydantic.dev/1.10/usage/validators/",
                "--embedding-provider",
                "mock",
            ]
        )
        assert code == 1
        assert not mock_client.called


# 13. Text-Only --dry-run Does Not Load Model or Instantiate MongoClient
def test_cli_dry_run_text_only_isolation() -> None:
    sample_file = FIXTURES_DIR / "pydantic_v1_sample.md"
    with patch("app.database.create_mongo_client") as mock_client, \
         patch("app.embeddings.get_embedding_provider") as mock_provider:
        code = cli_main(
            [
                "--source-file",
                str(sample_file),
                "--version",
                "1.10",
                "--url",
                "https://docs.pydantic.dev/1.10/usage/validators/",
                "--dry-run",
            ]
        )
        assert code == 0
        assert not mock_client.called
        assert not mock_provider.called


# 14. --dry-run-embeddings Does Not Instantiate MongoClient
def test_cli_dry_run_embeddings_isolation() -> None:
    sample_file = FIXTURES_DIR / "pydantic_v1_sample.md"
    with patch("app.database.create_mongo_client") as mock_client:
        code = cli_main(
            [
                "--source-file",
                str(sample_file),
                "--version",
                "1.10",
                "--url",
                "https://docs.pydantic.dev/1.10/usage/validators/",
                "--dry-run-embeddings",
                "--embedding-provider",
                "mock",
            ]
        )
        assert code == 0
        assert not mock_client.called


# 15. Vector Search Filters by Library and Version and Preserves Provenance
def test_retrieval_pipeline_filters_and_provenance() -> None:
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col

    synthetic_hit = {
        "chunk_id": "chk_target",
        "library": "pydantic",
        "version": "1.10",
        "canonical_url": "https://docs.pydantic.dev/1.10/validators/",
        "page_title": "Validators",
        "heading_path": "Validators > Custom Validators",
        "chunk_index": 1,
        "content": "Pydantic V1 uses @validator.",
        "char_count": 28,
        "score": 0.92,
    }
    mock_col.aggregate.return_value = [synthetic_hit]

    settings = Settings(app_env="test", embedding_provider="mock")
    query = RetrievalQuery(query="custom validator", library="pydantic", version="1.10", top_k=3)

    resp = perform_vector_search(db=mock_db, query=query, settings=settings)
    assert resp.total_results == 1
    hit = resp.results[0]
    assert hit.chunk_id == "chk_target"
    assert hit.canonical_url == "https://docs.pydantic.dev/1.10/validators/"
    assert hit.heading_path == "Validators > Custom Validators"
    assert hit.score == 0.92

    # Verify aggregation pipeline
    pipeline = mock_col.aggregate.call_args[0][0]
    vector_search_stage = pipeline[0]["$vectorSearch"]
    assert vector_search_stage["filter"]["$and"] == [
        {"library": {"$eq": "pydantic"}},
        {"version": {"$eq": "1.10"}},
    ]
    assert vector_search_stage["limit"] == 3
    # Check projection excludes embedding
    proj_stage = pipeline[1]["$project"]
    assert "embedding" not in proj_stage
    assert proj_stage["score"] == {"$meta": "vectorSearchScore"}


# 16. Mock Provider Prohibited in Non-Test Environments
def test_retrieval_endpoint_prohibits_mock_outside_test() -> None:
    mock_db = MagicMock()
    settings = Settings(app_env="production", embedding_provider="mock")
    query = RetrievalQuery(query="custom validator", version="1.10")

    with pytest.raises(VectorSearchUnavailableError, match="Mock embedding provider is prohibited"):
        perform_vector_search(db=mock_db, query=query, settings=settings)


# 17. Missing Index or Database Failure Returns 503
def test_retrieval_missing_index_returns_503() -> None:
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.__getitem__.return_value = mock_db
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col

    # Simulate Atlas Vector Search index missing error
    mock_col.aggregate.side_effect = OperationFailure("PlanExecutor error: vector search index not found")

    with TestClient(main_app) as client:
        client.app.state.mongo_client = mock_client
        with patch("app.main.settings", Settings(app_env="test", embedding_provider="mock")):
            resp = client.post(
                "/api/v1/retrieval/search",
                json={"query": "test query", "version": "1.10"},
            )
            assert resp.status_code == 503
            data = resp.json()
            assert data["detail"] == "Vector search service unavailable."
            # Verify no credentials or internal plan executor details leaked
            assert "PlanExecutor" not in data["detail"]


# 18. Missing Local Model Fails Without Network Download
def test_missing_local_model_fails_without_download() -> None:
    provider = SentenceTransformerEmbeddingProvider(
        model_name="non_existent_local_model_path",
        expected_dim=384,
    )
    mock_st_class = MagicMock(side_effect=OSError("Local files not found and automatic download prohibited."))
    mock_module = MagicMock(SentenceTransformer=mock_st_class)
    with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
        with pytest.raises(EmbeddingModelUnavailableError, match="Automatic downloads are prohibited"):
            provider.embed_text("sample query text")


# 19. Missing Dependency Fails Safely
def test_missing_dependency_fails_safely() -> None:
    provider = SentenceTransformerEmbeddingProvider(model_name="test_model", expected_dim=384)
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        with pytest.raises(EmbeddingDependencyMissingError, match="Package 'sentence-transformers' is not installed"):
            provider.embed_text("sample query")


# 20. HTTP API Endpoint Validation
def test_retrieval_endpoint_http_validation() -> None:
    mock_client = MagicMock()
    with TestClient(main_app) as client:
        client.app.state.mongo_client = mock_client
        # Empty query
        r1 = client.post("/api/v1/retrieval/search", json={"query": "   ", "version": "1.10"})
        assert r1.status_code == 422

        # Invalid version
        r2 = client.post("/api/v1/retrieval/search", json={"query": "valid query", "version": "3.0"})
        assert r2.status_code == 422

        # Invalid top_k
        r3 = client.post("/api/v1/retrieval/search", json={"query": "valid query", "version": "1.10", "top_k": 0})
        assert r3.status_code == 422

