import hashlib
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, UpdateOne
from pymongo.database import Database
from pymongo.errors import BulkWriteError, PyMongoError

from app.schemas import DocumentChunk, IngestionBatchSummary

logger = logging.getLogger(__name__)


class EmptySourceDocumentError(ValueError):
    """Raised when an empty or whitespace-only source is processed without explicit opt-in."""


def ensure_document_indexes(db: Database[Any]) -> None:
    """Ensure required indexes on document_chunks collection.

    Creates:
    1. Unique index on chunk_id.
    2. Compound index on (library, version, canonical_url).
    """
    collection = db["document_chunks"]
    try:
        collection.create_index([("chunk_id", ASCENDING)], unique=True, name="uniq_chunk_id")
        collection.create_index(
            [("library", ASCENDING), ("version", ASCENDING), ("canonical_url", ASCENDING)],
            name="idx_library_version_url",
        )
    except PyMongoError:
        logger.warning("MongoDB index creation failed.")
        raise


def _split_oversized_text(text: str, max_chunk_chars: int) -> list[str]:
    """Split text exceeding max_chunk_chars along paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    sub_chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_chunk and (current_len + para_len + 2 > max_chunk_chars):
            sub_chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_len = para_len
        else:
            current_chunk.append(para)
            current_len += para_len + 2

    if current_chunk:
        sub_chunks.append("\n\n".join(current_chunk))

    return sub_chunks


def chunk_markdown(
    text: str,
    canonical_url: str,
    page_title: str,
    version: str,
    library: str = "pydantic",
    source_release: str | None = None,
    source_commit: str | None = None,
    allow_empty_source: bool = False,
    max_chunk_chars: int = 1200,
) -> list[DocumentChunk]:
    """Parse Markdown text into heading-aware DocumentChunks deterministically.

    Preserves heading breadcrumbs, enforces character limits, and validates every chunk.
    """
    if not text or not text.strip():
        if not allow_empty_source:
            raise EmptySourceDocumentError("Source document is empty or whitespace-only.")
        return []

    lines = text.splitlines()
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

    sections: list[tuple[list[str], list[str]]] = []
    current_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []

    for line in lines:
        match = heading_pattern.match(line)
        if match:
            # If we had accumulated content before this heading, record it
            if current_lines:
                active_headings = [h[1] for h in current_stack] if current_stack else [page_title]
                sections.append((active_headings, current_lines))
                current_lines = []

            level = len(match.group(1))
            heading_text = match.group(2).strip()

            # Pop deeper or equal level headings from the hierarchy stack
            while current_stack and current_stack[-1][0] >= level:
                current_stack.pop()
            current_stack.append((level, heading_text))
        else:
            current_lines.append(line)

    if current_lines:
        active_headings = [h[1] for h in current_stack] if current_stack else [page_title]
        sections.append((active_headings, current_lines))

    chunks: list[DocumentChunk] = []
    global_chunk_index = 0

    for headings, sec_lines in sections:
        section_text = "\n".join(sec_lines).strip()
        if not section_text:
            continue

        heading_path = " > ".join(headings)

        # Handle splitting if section exceeds maximum character size
        if len(section_text) > max_chunk_chars:
            raw_splits = _split_oversized_text(section_text, max_chunk_chars)
        else:
            raw_splits = [section_text]

        for split_text in raw_splits:
            if not split_text.strip():
                continue

            content = split_text.strip()
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            # Deterministic chunk ID based on structural coordinates
            id_material = f"{library}:{version}:{canonical_url}:{heading_path}:{global_chunk_index}"
            chunk_id = hashlib.sha256(id_material.encode("utf-8")).hexdigest()

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                library=library,
                version=version,
                canonical_url=canonical_url,
                page_title=page_title,
                heading_path=heading_path,
                chunk_index=global_chunk_index,
                content=content,
                content_hash=content_hash,
                char_count=len(content),
                source_release=source_release,
                source_commit=source_commit,
            )
            chunks.append(chunk)
            global_chunk_index += 1

    return chunks


def _is_valid_vector(vec: Any, expected_dim: int = 384) -> bool:
    """Check if a vector is a non-empty list of finite floats matching expected dimension."""
    if not isinstance(vec, list) or len(vec) != expected_dim:
        return False
    for x in vec:
        if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x):
            return False
    return True


def store_and_reconcile_chunks(
    db: Database[Any],
    chunks: list[DocumentChunk],
    library: str,
    version: str,
    canonical_url: str,
    allow_empty_source: bool = False,
) -> IngestionBatchSummary:
    """Idempotently persist chunks and reconcile stale chunks for an exact document tuple."""
    try:
        ensure_document_indexes(db)
    except PyMongoError:
        return IngestionBatchSummary(
            library=library,
            version=version,
            source_file_or_url=canonical_url,
            success=False,
            error_message="MongoDB index creation or write failure during ingestion.",
        )

    collection = db["document_chunks"]

    # 1. Fetch existing chunks for the exact scoped tuple, including embedding for local validation
    try:
        existing_cursor = collection.find(
            {"library": library, "version": version, "canonical_url": canonical_url},
            {
                "chunk_id": 1,
                "content_hash": 1,
                "heading_path": 1,
                "page_title": 1,
                "source_release": 1,
                "embedding": 1,
            },
        )
        existing_map: dict[str, dict[str, Any]] = {
            doc["chunk_id"]: {
                "content_hash": doc.get("content_hash"),
                "heading_path": doc.get("heading_path"),
                "page_title": doc.get("page_title"),
                "source_release": doc.get("source_release"),
                "has_valid_vector": _is_valid_vector(doc.get("embedding")),
            }
            for doc in existing_cursor
        }
    except PyMongoError:
        logger.warning("MongoDB query failure while retrieving existing chunks.")
        return IngestionBatchSummary(
            library=library,
            version=version,
            source_file_or_url=canonical_url,
            success=False,
            error_message="MongoDB query failure while retrieving existing chunks.",
        )

    # 2. Diffing operations
    now = datetime.now(timezone.utc)
    operations: list[UpdateOne] = []
    chunks_skipped = 0
    intent_new = 0
    intent_update = 0

    for chunk in chunks:
        existing_meta = existing_map.get(chunk.chunk_id)
        if existing_meta is not None:
            # Preserve existing 4-field metadata comparison (source_commit remains deferred)
            is_metadata_unchanged = (
                existing_meta.get("content_hash") == chunk.content_hash
                and existing_meta.get("heading_path") == chunk.heading_path
                and existing_meta.get("page_title") == chunk.page_title
                and existing_meta.get("source_release") == chunk.source_release
            )
            has_valid_vector = existing_meta.get("has_valid_vector", False)

            if chunk.embedding is not None:
                if not _is_valid_vector(chunk.embedding):
                    return IngestionBatchSummary(
                        library=library,
                        version=version,
                        source_file_or_url=canonical_url,
                        success=False,
                        error_message="Incoming chunk has malformed embedding vector.",
                    )

                if is_metadata_unchanged and has_valid_vector:
                    # Content, metadata, and existing vector are all valid and unchanged: skip
                    chunks_skipped += 1
                    continue

                if is_metadata_unchanged and not has_valid_vector:
                    # Content and metadata match, but vector is missing or invalid: backfill!
                    operations.append(
                        UpdateOne(
                            {"chunk_id": chunk.chunk_id},
                            {"$set": {"embedding": chunk.embedding, "updated_at": now}},
                        )
                    )
                    intent_update += 1
                    continue

                # Content or metadata changed with valid new vector: atomic update
                payload = chunk.model_dump(exclude={"created_at", "updated_at"})
                operations.append(
                    UpdateOne(
                        {"chunk_id": chunk.chunk_id},
                        {"$set": {**payload, "updated_at": now}},
                    )
                )
                intent_update += 1
            else:
                # Incoming chunk has no embedding (embedding is None)
                if is_metadata_unchanged:
                    # Content and metadata match, and no incoming embedding to backfill: skip
                    chunks_skipped += 1
                    continue

                # Content or metadata changed!
                # Safety rule: A changed chunk must NEVER retain an existing stale vector for obsolete content.
                if has_valid_vector:
                    return IngestionBatchSummary(
                        library=library,
                        version=version,
                        source_file_or_url=canonical_url,
                        success=False,
                        error_message="Changed chunk lacks newly generated embedding vector. Aborting to prevent stale vector retention.",
                    )

                # Neither existing chunk nor incoming chunk has vector: update content/metadata
                payload = chunk.model_dump(exclude={"created_at", "updated_at"})
                operations.append(
                    UpdateOne(
                        {"chunk_id": chunk.chunk_id},
                        {"$set": {**payload, "updated_at": now}},
                    )
                )
                intent_update += 1
        else:
            # New chunk: queue upsert with created_at set strictly on insert
            if chunk.embedding is not None and not _is_valid_vector(chunk.embedding):
                return IngestionBatchSummary(
                    library=library,
                    version=version,
                    source_file_or_url=canonical_url,
                    success=False,
                    error_message="New chunk has malformed embedding vector.",
                )
            payload = chunk.model_dump(exclude={"created_at", "updated_at"})
            operations.append(
                UpdateOne(
                    {"chunk_id": chunk.chunk_id},
                    {
                        "$set": {**payload, "updated_at": now},
                        "$setOnInsert": {"created_at": now},
                    },
                    upsert=True,
                )
            )
            intent_new += 1

    chunks_inserted = 0
    chunks_updated = 0

    # 3. Execute bulk write if any updates/inserts are required
    if operations:
        try:
            bulk_res = collection.bulk_write(operations, ordered=True)
            chunks_inserted = bulk_res.upserted_count
            chunks_updated = bulk_res.modified_count
        except BulkWriteError as bwe:
            logger.warning("MongoDB bulk write failure during document persistence.")
            # Do not guess or fabricate counts; report partial failure
            details = getattr(bwe, "details", {})
            partial_ins = details.get("nUpserted", 0) if isinstance(details, dict) else 0
            partial_upd = details.get("nModified", 0) if isinstance(details, dict) else 0
            return IngestionBatchSummary(
                library=library,
                version=version,
                source_file_or_url=canonical_url,
                chunks_inserted=partial_ins,
                chunks_updated=partial_upd,
                chunks_skipped=chunks_skipped,
                chunks_deleted=0,
                success=False,
                error_message="MongoDB bulk write failure during document persistence.",
                partial_write=True,
            )
        except PyMongoError:
            logger.warning("MongoDB general write failure during document persistence.")
            return IngestionBatchSummary(
                library=library,
                version=version,
                source_file_or_url=canonical_url,
                chunks_inserted=0,
                chunks_updated=0,
                chunks_skipped=chunks_skipped,
                chunks_deleted=0,
                success=False,
                error_message="MongoDB general write failure during document persistence.",
                partial_write=True,
            )
    else:
        # All chunks skipped or no operations needed
        chunks_inserted = 0
        chunks_updated = 0

    # 4. Stale-chunk reconciliation (only executed on 100% write success)
    chunks_deleted = 0
    if not chunks and not allow_empty_source:
        # Safety guard: Never delete existing chunks for empty input without explicit opt-in
        pass
    else:
        active_ids = {c.chunk_id for c in chunks}
        stale_ids = set(existing_map.keys()) - active_ids
        if stale_ids:
            try:
                del_res = collection.delete_many(
                    {
                        "library": library,
                        "version": version,
                        "canonical_url": canonical_url,
                        "chunk_id": {"$in": list(stale_ids)},
                    }
                )
                chunks_deleted = del_res.deleted_count
            except PyMongoError:
                logger.warning("MongoDB deletion failure during stale chunk reconciliation.")
                return IngestionBatchSummary(
                    library=library,
                    version=version,
                    source_file_or_url=canonical_url,
                    chunks_inserted=chunks_inserted,
                    chunks_updated=chunks_updated,
                    chunks_skipped=chunks_skipped,
                    chunks_deleted=0,
                    success=False,
                    error_message="MongoDB deletion failure during stale chunk reconciliation.",
                    partial_write=True,
                )

    return IngestionBatchSummary(
        library=library,
        version=version,
        source_file_or_url=canonical_url,
        chunks_inserted=chunks_inserted,
        chunks_updated=chunks_updated,
        chunks_skipped=chunks_skipped,
        chunks_deleted=chunks_deleted,
        success=True,
    )

