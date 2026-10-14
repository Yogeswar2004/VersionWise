import logging
from typing import Any

from pymongo.database import Database
from pymongo.errors import PyMongoError

from app.config import Settings
from app.embeddings import EmbeddingProvider, get_embedding_provider, validate_numeric_vector
from app.schemas import RetrievalQuery, RetrievalResponse, RetrievalResult

logger = logging.getLogger(__name__)


class VectorSearchUnavailableError(Exception):
    """Raised when vector search is unavailable, misconfigured, or fails."""


def build_vector_search_pipeline(
    index_name: str,
    query_vector: list[float],
    library: str,
    version: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Build the MongoDB Atlas $vectorSearch aggregation pipeline with library/version pre-filtering."""
    return [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": max(top_k * 10, 50),
                "limit": top_k,
                "filter": {
                    "$and": [
                        {"library": {"$eq": library}},
                        {"version": {"$eq": version}},
                    ]
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "chunk_id": 1,
                "library": 1,
                "version": 1,
                "canonical_url": 1,
                "page_title": 1,
                "heading_path": 1,
                "chunk_index": 1,
                "content": 1,
                "char_count": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]


def perform_vector_search(
    db: Database[Any],
    query: RetrievalQuery,
    settings: Settings,
    provider: EmbeddingProvider | None = None,
) -> RetrievalResponse:
    """Execute version-aware vector retrieval against MongoDB Atlas document_chunks collection."""
    # Safety guard: prevent mock vectors from serving retrieval in production/staging environments
    if settings.app_env != "test" and settings.embedding_provider == "mock":
        logger.warning("Attempted to use mock embedding provider in non-test environment.")
        raise VectorSearchUnavailableError("Mock embedding provider is prohibited in non-test environments.")

    # Generate and validate query embedding
    if provider is None:
        provider = get_embedding_provider(settings)

    try:
        raw_vector = provider.embed_text(query.query)
        query_vector = validate_numeric_vector(
            raw_vector,
            expected_dim=settings.embedding_dimensions,
            check_normalized=True,
        )
    except Exception as exc:
        logger.warning("Query embedding generation or validation failed.")
        raise VectorSearchUnavailableError("Failed to generate query embedding.") from exc

    pipeline = build_vector_search_pipeline(
        index_name=settings.vector_search_index_name,
        query_vector=query_vector,
        library=query.library,
        version=query.version,
        top_k=query.top_k,
    )

    try:
        collection = db["document_chunks"]
        cursor = collection.aggregate(pipeline)
        results: list[RetrievalResult] = []
        for doc in cursor:
            score = float(doc.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            results.append(
                RetrievalResult(
                    chunk_id=doc["chunk_id"],
                    library=doc["library"],
                    version=doc["version"],
                    canonical_url=doc["canonical_url"],
                    page_title=doc["page_title"],
                    heading_path=doc["heading_path"],
                    chunk_index=doc["chunk_index"],
                    content=doc["content"],
                    char_count=doc["char_count"],
                    score=score,
                )
            )

        return RetrievalResponse(
            query=query.query,
            library=query.library,
            version=query.version,
            total_results=len(results),
            results=results,
        )
    except PyMongoError:
        logger.warning("MongoDB vector search failed: operation or index error.")
        raise VectorSearchUnavailableError("Vector search service unavailable.")

