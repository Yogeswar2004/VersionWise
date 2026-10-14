from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from pymongo.errors import BulkWriteError, PyMongoError

from app.ingestion import (
    EmptySourceDocumentError,
    chunk_markdown,
    ensure_document_indexes,
    store_and_reconcile_chunks,
)
from app.schemas import DocumentChunk
from scripts.ingest_docs import generate_canonical_url, main as cli_main

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_document_chunk_version_validation() -> None:
    # Valid versions
    c1 = DocumentChunk(
        chunk_id="abc1",
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/",
        page_title="Title",
        heading_path="Title",
        chunk_index=0,
        content="Valid content",
        content_hash="hash1",
        char_count=13,
    )
    assert c1.version == "1.10"

    c2 = DocumentChunk(
        chunk_id="abc2",
        library="pydantic",
        version="2.12",
        canonical_url="https://docs.pydantic.dev/2.12/",
        page_title="Title",
        heading_path="Title",
        chunk_index=0,
        content="Valid content",
        content_hash="hash2",
        char_count=13,
    )
    assert c2.version == "2.12"

    # Invalid versions
    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="abc3",
            library="pydantic",
            version="3.0",
            canonical_url="https://docs.pydantic.dev/3.0/",
            page_title="Title",
            heading_path="Title",
            chunk_index=0,
            content="Valid content",
            content_hash="hash3",
            char_count=13,
        )

    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="abc4",
            library="pydantic",
            version="latest",
            canonical_url="https://docs.pydantic.dev/latest/",
            page_title="Title",
            heading_path="Title",
            chunk_index=0,
            content="Valid content",
            content_hash="hash4",
            char_count=13,
        )

    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="abc5",
            library="pydantic",
            version="v1.10",
            canonical_url="https://docs.pydantic.dev/1.10/",
            page_title="Title",
            heading_path="Title",
            chunk_index=0,
            content="Valid content",
            content_hash="hash5",
            char_count=13,
        )


def test_deterministic_chunking_and_ids() -> None:
    text = "# Models\n\nBasic models content here."
    chunks1 = chunk_markdown(
        text=text,
        canonical_url="https://docs.pydantic.dev/1.10/models/",
        page_title="Models",
        version="1.10",
    )
    chunks2 = chunk_markdown(
        text=text,
        canonical_url="https://docs.pydantic.dev/1.10/models/",
        page_title="Models",
        version="1.10",
    )

    assert len(chunks1) == 1
    assert len(chunks2) == 1
    assert chunks1[0].chunk_id == chunks2[0].chunk_id
    assert chunks1[0].content_hash == chunks2[0].content_hash
    assert chunks1[0].version == "1.10"


def test_heading_breadcrumbs_from_fixture() -> None:
    v1_fixture = (FIXTURES_DIR / "pydantic_v1_sample.md").read_text(encoding="utf-8")
    chunks = chunk_markdown(
        text=v1_fixture,
        canonical_url="https://docs.pydantic.dev/1.10/validators/",
        page_title="Validators",
        version="1.10",
    )

    assert len(chunks) >= 2
    paths = [c.heading_path for c in chunks]
    assert any("Custom Validators" in p for p in paths)
    assert any("Settings Management" in p for p in paths)
    for c in chunks:
        assert c.version == "1.10"
        assert c.library == "pydantic"


def test_oversized_section_splitting() -> None:
    # Create section with three 600-character paragraphs (> 1200 total)
    para1 = "A" * 600
    para2 = "B" * 600
    para3 = "C" * 600
    text = f"# Section\n\n{para1}\n\n{para2}\n\n{para3}"

    chunks = chunk_markdown(
        text=text,
        canonical_url="https://docs.pydantic.dev/1.10/long/",
        page_title="Long",
        version="1.10",
        max_chunk_chars=1000,
    )

    assert len(chunks) >= 2
    # Verify sequential chunk indexing and shared heading breadcrumb
    assert chunks[0].heading_path == "Section"
    assert chunks[1].heading_path == "Section"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_empty_and_whitespace_source_handling() -> None:
    with pytest.raises(EmptySourceDocumentError):
        chunk_markdown(
            text="",
            canonical_url="https://docs.pydantic.dev/1.10/empty/",
            page_title="Empty",
            version="1.10",
        )

    with pytest.raises(EmptySourceDocumentError):
        chunk_markdown(
            text="   \n\n  \t  ",
            canonical_url="https://docs.pydantic.dev/1.10/empty/",
            page_title="Empty",
            version="1.10",
        )

    # Opt-in allow_empty_source
    res = chunk_markdown(
        text="",
        canonical_url="https://docs.pydantic.dev/1.10/empty/",
        page_title="Empty",
        version="1.10",
        allow_empty_source=True,
    )
    assert res == []


def test_ensure_document_indexes() -> None:
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col

    ensure_document_indexes(mock_db)

    assert mock_col.create_index.call_count == 2
    calls = mock_col.create_index.call_args_list
    assert calls[0][1]["unique"] is True
    assert calls[0][1]["name"] == "uniq_chunk_id"
    assert calls[1][1]["name"] == "idx_library_version_url"


def test_new_chunk_insertion_and_timestamps() -> None:
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col
    mock_col.find.return_value = []  # No existing chunks

    bulk_res = MagicMock()
    bulk_res.upserted_count = 1
    bulk_res.modified_count = 0
    mock_col.bulk_write.return_value = bulk_res

    chunk = DocumentChunk(
        chunk_id="chunk1",
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
        page_title="Test",
        heading_path="Test",
        chunk_index=0,
        content="Sample content",
        content_hash="hash1",
        char_count=14,
    )

    summary = store_and_reconcile_chunks(
        db=mock_db,
        chunks=[chunk],
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
    )

    assert summary.success is True
    assert summary.chunks_inserted == 1
    assert summary.chunks_skipped == 0
    assert summary.chunks_updated == 0

    # Verify operation structure
    op = mock_col.bulk_write.call_args[0][0][0]
    update_doc = op._doc
    assert "$setOnInsert" in update_doc
    assert "created_at" in update_doc["$setOnInsert"]
    assert "$set" in update_doc
    assert "updated_at" in update_doc["$set"]
    assert "created_at" not in update_doc["$set"]


def test_unchanged_chunk_skipping() -> None:
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col

    # Existing chunk matches content_hash, heading_path, page_title, source_release
    mock_col.find.return_value = [
        {
            "chunk_id": "chunk1",
            "content_hash": "hash1",
            "heading_path": "Test",
            "page_title": "Test",
            "source_release": None,
        }
    ]

    chunk = DocumentChunk(
        chunk_id="chunk1",
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
        page_title="Test",
        heading_path="Test",
        chunk_index=0,
        content="Sample content",
        content_hash="hash1",
        char_count=14,
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
    assert summary.chunks_inserted == 0
    assert summary.chunks_updated == 0
    assert not mock_col.bulk_write.called


def test_content_and_metadata_update_detection() -> None:
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col

    # Case A: content_hash differs
    mock_col.find.return_value = [
        {
            "chunk_id": "chunk1",
            "content_hash": "old_hash",
            "heading_path": "Test",
            "page_title": "Test",
            "source_release": None,
        }
    ]

    bulk_res = MagicMock()
    bulk_res.upserted_count = 0
    bulk_res.modified_count = 1
    mock_col.bulk_write.return_value = bulk_res

    chunk = DocumentChunk(
        chunk_id="chunk1",
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
        page_title="Test",
        heading_path="Test",
        chunk_index=0,
        content="Modified content",
        content_hash="new_hash",
        char_count=16,
    )

    summary = store_and_reconcile_chunks(
        db=mock_db,
        chunks=[chunk],
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
    )

    assert summary.chunks_updated == 1
    assert summary.chunks_skipped == 0
    op = mock_col.bulk_write.call_args[0][0][0]
    assert "$setOnInsert" not in op._doc  # Update only; not upsert
    assert "$set" in op._doc


def test_metadata_only_update_preserves_created_at() -> None:
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col

    # Existing chunk has IDENTICAL content_hash, but different page_title
    mock_col.find.return_value = [
        {
            "chunk_id": "chunk1",
            "content_hash": "hash1",
            "heading_path": "Test",
            "page_title": "Old Page Title",
            "source_release": None,
        }
    ]

    bulk_res = MagicMock()
    bulk_res.upserted_count = 0
    bulk_res.modified_count = 1
    mock_col.bulk_write.return_value = bulk_res

    chunk = DocumentChunk(
        chunk_id="chunk1",
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
        page_title="New Page Title",
        heading_path="Test",
        chunk_index=0,
        content="Sample content",
        content_hash="hash1",
        char_count=14,
    )

    summary = store_and_reconcile_chunks(
        db=mock_db,
        chunks=[chunk],
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
    )

    assert summary.chunks_updated == 1
    assert summary.chunks_skipped == 0
    assert mock_col.bulk_write.called
    op = mock_col.bulk_write.call_args[0][0][0]
    # Verify update payload contains new page_title and updated_at
    assert op._doc["$set"]["page_title"] == "New Page Title"
    assert "updated_at" in op._doc["$set"]
    # Verify created_at is NOT overwritten
    assert "$setOnInsert" not in op._doc


def test_stale_chunk_reconciliation() -> None:
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col

    # Existing chunks are chunk1 and chunk2
    mock_col.find.return_value = [
        {
            "chunk_id": "chunk1",
            "content_hash": "hash1",
            "heading_path": "Test",
            "page_title": "Test",
            "source_release": None,
        },
        {
            "chunk_id": "chunk2",
            "content_hash": "hash2",
            "heading_path": "Test > Removed",
            "page_title": "Test",
            "source_release": None,
        },
    ]

    del_res = MagicMock()
    del_res.deleted_count = 1
    mock_col.delete_many.return_value = del_res

    # New chunks only contain chunk1
    chunk1 = DocumentChunk(
        chunk_id="chunk1",
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
        page_title="Test",
        heading_path="Test",
        chunk_index=0,
        content="Sample content",
        content_hash="hash1",
        char_count=14,
    )

    summary = store_and_reconcile_chunks(
        db=mock_db,
        chunks=[chunk1],
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
    )

    assert summary.chunks_deleted == 1
    mock_col.delete_many.assert_called_once_with(
        {
            "library": "pydantic",
            "version": "1.10",
            "canonical_url": "https://docs.pydantic.dev/1.10/test/",
            "chunk_id": {"$in": ["chunk2"]},
        }
    )


def test_no_stale_deletion_on_bulk_write_error() -> None:
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col
    mock_col.find.return_value = [{"chunk_id": "chunk2", "content_hash": "h2"}]
    mock_col.bulk_write.side_effect = PyMongoError("Connection timed out")

    chunk1 = DocumentChunk(
        chunk_id="chunk1",
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
        page_title="Test",
        heading_path="Test",
        chunk_index=0,
        content="Sample content",
        content_hash="hash1",
        char_count=14,
    )

    summary = store_and_reconcile_chunks(
        db=mock_db,
        chunks=[chunk1],
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
    )

    assert summary.success is False
    assert summary.partial_write is True
    assert not mock_col.delete_many.called
    assert "MongoDB general write failure" in summary.error_message


def test_partial_bulk_write_reporting() -> None:
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col
    mock_col.find.return_value = []

    bwe = BulkWriteError({"nUpserted": 2, "nModified": 1, "writeErrors": []})
    mock_col.bulk_write.side_effect = bwe

    chunk1 = DocumentChunk(
        chunk_id="chunk1",
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
        page_title="Test",
        heading_path="Test",
        chunk_index=0,
        content="Sample content",
        content_hash="hash1",
        char_count=14,
    )

    summary = store_and_reconcile_chunks(
        db=mock_db,
        chunks=[chunk1],
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
    )

    assert summary.success is False
    assert summary.partial_write is True
    assert summary.chunks_inserted == 2
    assert summary.chunks_updated == 1
    assert not mock_col.delete_many.called


def test_index_creation_failure_aborts_writes_and_reconciliation() -> None:
    mock_db = MagicMock()
    mock_col = MagicMock()
    mock_db.__getitem__.return_value = mock_col
    mock_col.create_index.side_effect = PyMongoError("Simulated index collision error")

    chunk = DocumentChunk(
        chunk_id="chunk1",
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
        page_title="Test",
        heading_path="Test",
        chunk_index=0,
        content="Sample content",
        content_hash="hash1",
        char_count=14,
    )

    summary = store_and_reconcile_chunks(
        db=mock_db,
        chunks=[chunk],
        library="pydantic",
        version="1.10",
        canonical_url="https://docs.pydantic.dev/1.10/test/",
    )

    assert summary.success is False
    assert summary.error_message == "MongoDB index creation or write failure during ingestion."
    assert not mock_col.bulk_write.called
    assert not mock_col.delete_many.called


def test_canonical_url_generation_algorithm() -> None:
    base = "https://docs.pydantic.dev/1.10/"
    # Normal distinct pages
    assert generate_canonical_url(base, "models.md") == "https://docs.pydantic.dev/1.10/models/"
    assert generate_canonical_url(base, "usage/types.md") == "https://docs.pydantic.dev/1.10/usage/types/"

    # Index and README files normalize to directory URLs
    assert generate_canonical_url(base, "index.md") == "https://docs.pydantic.dev/1.10/"
    assert generate_canonical_url(base, "models/index.md") == "https://docs.pydantic.dev/1.10/models/"
    assert generate_canonical_url(base, "models/README.md") == "https://docs.pydantic.dev/1.10/models/"


def test_dry_run_isolation() -> None:
    sample_file = FIXTURES_DIR / "pydantic_v1_sample.md"

    with patch("app.database.create_mongo_client") as mock_create_client:
        code = cli_main(
            [
                "--source-file",
                str(sample_file),
                "--version",
                "1.10",
                "--url",
                "https://docs.pydantic.dev/1.10/sample/",
                "--dry-run",
            ]
        )
        assert code == 0
        assert not mock_create_client.called


def test_cli_directory_dry_run_multiple_files(tmp_path: Path) -> None:
    file1 = tmp_path / "models.md"
    file1.write_text("# Models\n\nContent for models.", encoding="utf-8")
    file2 = tmp_path / "types.md"
    file2.write_text("# Types\n\nContent for types.", encoding="utf-8")

    with patch("app.database.create_mongo_client") as mock_client:
        code = cli_main(
            [
                "--source-dir",
                str(tmp_path),
                "--version",
                "1.10",
                "--url",
                "https://docs.pydantic.dev/1.10/",
                "--dry-run",
            ]
        )
        assert code == 0
        assert not mock_client.called


def test_cli_directory_duplicate_url_detection(tmp_path: Path) -> None:
    file1 = tmp_path / "models.md"
    file1.write_text("# Models\n\nContent A.", encoding="utf-8")
    sub_dir = tmp_path / "models"
    sub_dir.mkdir()
    file2 = sub_dir / "index.md"
    file2.write_text("# Models Index\n\nContent B.", encoding="utf-8")

    with patch("app.database.create_mongo_client") as mock_client:
        code = cli_main(
            [
                "--source-dir",
                str(tmp_path),
                "--version",
                "1.10",
                "--url",
                "https://docs.pydantic.dev/1.10/",
                "--dry-run",
            ]
        )
        assert code == 1
        assert not mock_client.called


def test_cli_directory_version_path_conflict(tmp_path: Path) -> None:
    v2_dir = tmp_path / "pydantic" / "2.12" / "docs"
    v2_dir.mkdir(parents=True)
    file1 = v2_dir / "models.md"
    file1.write_text("# Models\n\nContent V2.", encoding="utf-8")

    # Conflict: Directory explicitly contains '2.12', but --version 1.10 was passed
    code = cli_main(
        [
            "--source-dir",
            str(v2_dir),
            "--version",
            "1.10",
            "--url",
            "https://docs.pydantic.dev/1.10/",
            "--dry-run",
        ]
    )
    assert code == 1

    # Non-conflicting: Directory path has '2.12' and --version 2.12 matches
    code_ok = cli_main(
        [
            "--source-dir",
            str(v2_dir),
            "--version",
            "2.12",
            "--url",
            "https://docs.pydantic.dev/2.12/",
            "--dry-run",
        ]
    )
    assert code_ok == 0


def test_v2_sample_fixture_parsing() -> None:
    v2_fixture = (FIXTURES_DIR / "pydantic_v2_sample.md").read_text(encoding="utf-8")
    chunks = chunk_markdown(
        text=v2_fixture,
        canonical_url="https://docs.pydantic.dev/2.12/sample/",
        page_title="V2 Sample",
        version="2.12",
    )

    assert len(chunks) >= 2
    for c in chunks:
        assert c.version == "2.12"
        assert c.library == "pydantic"
    assert any("Field Validators" in c.heading_path for c in chunks)
    assert any("Settings Migration" in c.heading_path for c in chunks)
