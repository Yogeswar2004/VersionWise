import math
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class DocumentChunk(BaseModel):
    """Schema representing an extracted, heading-aware documentation chunk."""

    chunk_id: str = Field(min_length=1, description="Deterministic SHA-256 chunk identifier")
    library: str = Field(default="pydantic", min_length=1)
    version: str = Field(min_length=1, description="Product target version, strictly '1.10' or '2.12'")
    canonical_url: str = Field(min_length=1)
    page_title: str = Field(min_length=1)
    heading_path: str = Field(min_length=1, description="Hierarchical breadcrumb of section headers")
    chunk_index: int = Field(ge=0, description="Sequential index within the document section")
    content: str = Field(min_length=1, description="Text/markdown content of the chunk")
    content_hash: str = Field(min_length=1, description="SHA-256 hash of the chunk content")
    char_count: int = Field(ge=1, description="Character count of the content")
    source_release: str | None = Field(default=None, description="Exact upstream release tag (e.g. 'v1.10.19')")
    source_commit: str | None = Field(default=None, description="Exact upstream Git commit SHA")
    embedding: list[float] | None = Field(default=None, description="384-dimensional dense semantic embedding vector")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        clean_v = v.strip()
        if clean_v not in ("1.10", "2.12"):
            raise ValueError(f"Unsupported product version '{v}'. Allowed versions: '1.10', '2.12'.")
        return clean_v

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("Embedding must be a list of floats.")
        if len(v) != 384:
            raise ValueError(f"Embedding vector must have exactly 384 dimensions, got {len(v)}.")
        for i, val in enumerate(v):
            if isinstance(val, bool) or not isinstance(val, (float, int)):
                raise ValueError(f"Embedding component at index {i} must be a float or int, not boolean or other type.")
            if not math.isfinite(val):
                raise ValueError(f"Embedding component at index {i} must be a finite number (not NaN or Inf).")
        return [float(x) for x in v]


class IngestionBatchSummary(BaseModel):
    """Summary metrics of an ingestion run."""

    library: str
    version: str
    source_file_or_url: str
    chunks_inserted: int = 0
    chunks_updated: int = 0
    chunks_skipped: int = 0
    chunks_deleted: int = 0
    success: bool = True
    error_message: str | None = None
    partial_write: bool = False


class RetrievalQuery(BaseModel):
    """Request schema for version-aware documentation search."""

    query: str = Field(min_length=1, max_length=1000, description="Natural language search query")
    library: str = Field(default="pydantic", description="Target library")
    version: str = Field(description="Product target version, strictly '1.10' or '2.12'")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum number of chunks to retrieve")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        clean_v = v.strip()
        if not clean_v:
            raise ValueError("Query cannot be empty or whitespace-only.")
        if len(clean_v) > 1000:
            raise ValueError("Query cannot exceed 1000 characters.")
        return clean_v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        clean_v = v.strip()
        if clean_v not in ("1.10", "2.12"):
            raise ValueError(f"Unsupported product version '{v}'. Allowed versions: '1.10', '2.12'.")
        return clean_v

    @field_validator("library")
    @classmethod
    def validate_library(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v != "pydantic":
            raise ValueError(f"Unsupported library '{v}'. Currently supported library: 'pydantic'.")
        return clean_v


class RetrievalResult(BaseModel):
    """Individual retrieved documentation chunk with relevance score and provenance."""

    chunk_id: str
    library: str
    version: str
    canonical_url: str
    page_title: str
    heading_path: str
    chunk_index: int
    content: str
    char_count: int
    score: float = Field(ge=0.0, le=1.0, description="Cosine similarity score")


class RetrievalResponse(BaseModel):
    """Response schema containing ranked retrieval results."""

    query: str
    library: str
    version: str
    total_results: int
    results: list[RetrievalResult]

