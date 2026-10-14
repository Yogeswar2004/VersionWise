import hashlib
import math
import random
from typing import Any, Protocol, runtime_checkable

from app.config import Settings
from app.schemas import DocumentChunk


class EmbeddingError(Exception):
    """Base exception for embedding operations."""


class EmbeddingDependencyMissingError(EmbeddingError):
    """Raised when sentence-transformers or its dependencies are not installed."""


class EmbeddingModelUnavailableError(EmbeddingError):
    """Raised when the local model files cannot be loaded without network downloads."""


class EmbeddingModelConfigurationError(EmbeddingError):
    """Raised when the loaded model configuration or tokenizer cannot be reliably verified."""


def validate_numeric_vector(
    vec: Any,
    expected_dim: int = 384,
    check_normalized: bool = True,
    tolerance: float = 1e-3,
) -> list[float]:
    """Validate that a vector consists strictly of finite numbers and matches the expected dimension.

    Rejects booleans, non-numeric values, NaNs, infinities, wrong lengths, and zero-norm vectors.
    """
    if not isinstance(vec, list):
        raise ValueError(f"Vector must be a list, got {type(vec).__name__}.")
    if len(vec) != expected_dim:
        raise ValueError(f"Vector dimension mismatch: expected {expected_dim}, got {len(vec)}.")

    norm_sq = 0.0
    for i, val in enumerate(vec):
        if isinstance(val, bool) or not isinstance(val, (float, int)):
            raise ValueError(f"Vector component at index {i} must be a float or int, not boolean or other type.")
        if not math.isfinite(val):
            raise ValueError(f"Vector component at index {i} must be a finite number (not NaN or Inf).")
        norm_sq += float(val) * float(val)

    if check_normalized:
        norm = math.sqrt(norm_sq)
        if norm < 1e-6:
            raise ValueError("Vector has zero or near-zero norm; cannot be normalized.")
        if abs(norm - 1.0) > tolerance:
            raise ValueError(f"Vector is not unit normalized: norm is {norm:.6f}, expected ~1.0.")

    return [float(x) for x in vec]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol defining the embedding provider interface."""

    def embed_text(self, text: str) -> list[float]:
        """Generate a normalized embedding vector for a text string."""
        ...

    def embed_chunk(self, chunk: DocumentChunk) -> list[float]:
        """Generate an embedding vector for a DocumentChunk."""
        ...

    def embed_batch(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        """Generate embedding vectors for a list of DocumentChunks."""
        ...

    def get_dimension(self) -> int:
        """Return the vector dimension produced by this provider."""
        ...


class MockEmbeddingProvider:
    """Deterministic, zero-dependency embedding provider for offline tests and in-memory dry-runs.

    Generates reproducible 384-dimensional unit-normalized float vectors using SHA-256 seeding.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def get_dimension(self) -> int:
        return self.dim

    def _deterministic_unit_vector(self, text: str) -> list[float]:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot embed empty or whitespace-only text.")
        # Seed deterministic pseudo-random generator with text hash
        seed = int(hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        raw = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in raw))
        if norm < 1e-6:
            raw[0] = 1.0
            norm = 1.0
        unit_vec = [x / norm for x in raw]
        return validate_numeric_vector(unit_vec, expected_dim=self.dim, check_normalized=True)

    def embed_text(self, text: str) -> list[float]:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot embed empty or whitespace-only text.")
        # Simulate sub-window pooling if text is very long (over 1000 characters)
        if len(clean_text) > 1000:
            window_size = 500
            stride = 250
            sub_vecs: list[list[float]] = []
            for start in range(0, len(clean_text), stride):
                window = clean_text[start : start + window_size]
                if window.strip():
                    sub_vecs.append(self._deterministic_unit_vector(window))
                if start + window_size >= len(clean_text):
                    break
            if sub_vecs:
                mean_vec = [0.0] * self.dim
                for v in sub_vecs:
                    for d in range(self.dim):
                        mean_vec[d] += v[d]
                mean_vec = [x / len(sub_vecs) for x in mean_vec]
                norm = math.sqrt(sum(x * x for x in mean_vec))
                if norm < 1e-6:
                    mean_vec[0] = 1.0
                    norm = 1.0
                pooled = [x / norm for x in mean_vec]
                return validate_numeric_vector(pooled, expected_dim=self.dim, check_normalized=True)

        return self._deterministic_unit_vector(clean_text)

    def embed_chunk(self, chunk: DocumentChunk) -> list[float]:
        text = f"{chunk.heading_path}\n\n{chunk.content}"
        return self.embed_text(text)

    def embed_batch(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        return [self.embed_chunk(c) for c in chunks]


class SentenceTransformerEmbeddingProvider:
    """Embedding provider using local SentenceTransformer models with lazy initialization.

    Enforces local-only model loading, verifies pooling dimensions, and uses token-aware
    overlapping sub-windows with normalized mean pooling for passages exceeding model capacity.
    """

    def __init__(self, model_name: str, expected_dim: int = 384) -> None:
        self.model_name = model_name
        self.expected_dim = expected_dim
        self._model: Any = None
        self._tokenizer: Any = None
        self._max_seq_length: int = 256
        self._window_capacity: int = 254

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingDependencyMissingError(
                "Package 'sentence-transformers' is not installed. To run real vector embeddings, "
                "install sentence-transformers or configure EMBEDDING_PROVIDER=mock for offline tests."
            ) from exc

        try:
            # Strictly local loading: never download model weights or tokenizer files automatically
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception as exc:
            raise EmbeddingModelUnavailableError(
                f"Failed to load local model '{self.model_name}'. Automatic downloads are prohibited. "
                f"Ensure model files are cached locally."
            ) from exc

        actual_dim = self._model.get_sentence_embedding_dimension()
        if actual_dim != self.expected_dim:
            raise ValueError(
                f"Model dimension mismatch: loaded model '{self.model_name}' produces {actual_dim} dimensions, "
                f"but configuration requires {self.expected_dim}."
            )

        self._tokenizer = getattr(self._model, "tokenizer", None)
        max_seq = getattr(self._model, "max_seq_length", None)
        if not isinstance(max_seq, int) or max_seq <= 2 or self._tokenizer is None:
            raise EmbeddingModelConfigurationError(
                f"Unable to reliably verify sequence length and tokenizer for model '{self.model_name}'."
            )

        self._max_seq_length = max_seq
        try:
            num_special = len(self._tokenizer.build_inputs_with_special_tokens([]))
        except Exception:
            num_special = 2

        self._window_capacity = self._max_seq_length - num_special
        if self._window_capacity <= 0:
            raise EmbeddingModelConfigurationError(
                f"Invalid window capacity ({self._window_capacity}) for model '{self.model_name}'."
            )

    def get_dimension(self) -> int:
        return self.expected_dim

    def embed_text(self, text: str) -> list[float]:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot embed empty or whitespace-only text.")

        self._ensure_loaded()

        # Tokenize content without special tokens to evaluate sequence length
        token_ids = self._tokenizer.encode(clean_text, add_special_tokens=False)

        if len(token_ids) <= self._window_capacity:
            # Single window: fits within model sequence budget without truncation
            raw_vec = self._model.encode(clean_text, normalize_embeddings=True).tolist()
            return validate_numeric_vector(raw_vec, expected_dim=self.expected_dim, check_normalized=True)

        # Multi-window overlapping pooling
        stride = max(1, self._window_capacity // 2)
        window_vectors: list[list[float]] = []

        for start_idx in range(0, len(token_ids), stride):
            sub_ids = token_ids[start_idx : start_idx + self._window_capacity]
            if not sub_ids:
                break
            sub_text = self._tokenizer.decode(sub_ids, skip_special_tokens=True).strip()
            if not sub_text:
                continue
            sub_vec = self._model.encode(sub_text, normalize_embeddings=True).tolist()
            window_vectors.append(sub_vec)
            if start_idx + self._window_capacity >= len(token_ids):
                break

        if not window_vectors:
            raw_vec = self._model.encode(clean_text[:500], normalize_embeddings=True).tolist()
            return validate_numeric_vector(raw_vec, expected_dim=self.expected_dim, check_normalized=True)

        # Uniform mean pooling across sub-windows
        mean_vec = [0.0] * self.expected_dim
        for w_vec in window_vectors:
            for d in range(self.expected_dim):
                mean_vec[d] += w_vec[d]
        mean_vec = [x / len(window_vectors) for x in mean_vec]

        # Final L2 normalization
        norm = math.sqrt(sum(x * x for x in mean_vec))
        if norm < 1e-6:
            raise ValueError("Aggregated window vector has zero norm.")
        final_vec = [x / norm for x in mean_vec]

        return validate_numeric_vector(final_vec, expected_dim=self.expected_dim, check_normalized=True)

    def embed_chunk(self, chunk: DocumentChunk) -> list[float]:
        text = f"{chunk.heading_path}\n\n{chunk.content}"
        return self.embed_text(text)

    def embed_batch(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        return [self.embed_chunk(c) for c in chunks]


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Factory creating the configured embedding provider."""
    provider_name = settings.embedding_provider.strip().lower()
    if provider_name == "mock":
        return MockEmbeddingProvider(dim=settings.embedding_dimensions)
    if provider_name == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(
            model_name=settings.embedding_model_name,
            expected_dim=settings.embedding_dimensions,
        )
    raise ValueError(f"Unsupported embedding provider: '{settings.embedding_provider}'.")

