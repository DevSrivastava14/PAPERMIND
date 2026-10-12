from typing import Any, List, Optional
import numpy as np

from app.chunking.models import TextChunk
from app.embeddings.base import BaseEmbedder
from app.embeddings.models import EmbeddingResult


class SentenceTransformerEmbedder(BaseEmbedder):
    """Generates dense numerical vector embeddings using sentence-transformers models."""

    DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_DIMENSION: int = 384

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = 32,
        device: Optional[str] = None,
        model: Optional[Any] = None,
        dimension: Optional[int] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self._model_name = model_name
        self.default_batch_size = batch_size
        self.device = device
        self._dimension = dimension

        if model is not None:
            self._model = model
        else:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name, device=device)

    @property
    def model_name(self) -> str:
        """The identifier or name of the embedding model."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        if self._dimension is not None:
            return self._dimension
        if hasattr(self._model, "get_sentence_embedding_dimension"):
            dim = self._model.get_sentence_embedding_dimension()
            if dim is not None:
                self._dimension = int(dim)
                return self._dimension
        return self.DEFAULT_DIMENSION

    def embed_texts(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Embeds raw text strings into a 2D float32 NumPy array of shape (N, dimension)."""
        effective_batch_size = (
            batch_size if batch_size is not None else self.default_batch_size
        )
        if effective_batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {effective_batch_size}"
            )

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        batches: List[np.ndarray] = []
        for i in range(0, len(texts), effective_batch_size):
            batch_texts = texts[i : i + effective_batch_size]
            raw_output = self._model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            if hasattr(raw_output, "detach"):
                raw_output = raw_output.detach().cpu().numpy()

            batch_array = np.asarray(raw_output, dtype=np.float32)
            if batch_array.ndim == 1:
                batch_array = np.expand_dims(batch_array, axis=0)

            batches.append(batch_array)

        vectors = np.vstack(batches)
        self._validate_vectors(vectors, expected_count=len(texts))
        return vectors

    def _validate_vectors(self, vectors: np.ndarray, expected_count: int) -> None:
        """Validates that vectors are 2D NumPy float32 arrays, match expected dimensions, and are finite."""
        if not isinstance(vectors, np.ndarray):
            raise TypeError(
                f"Vectors must be a NumPy ndarray, got {type(vectors).__name__}"
            )

        if vectors.ndim != 2:
            raise ValueError(
                f"Vectors must be 2-dimensional (shape: [num_vectors, dimension]), got {vectors.ndim}D array with shape {vectors.shape}"
            )

        if vectors.shape[0] != expected_count:
            raise ValueError(
                f"Vector count mismatch: expected {expected_count} vectors, got {vectors.shape[0]}"
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected dimension {self.dimension}, got {vectors.shape[1]}"
            )

        if not np.all(np.isfinite(vectors)):
            raise ValueError("Embedding vectors contain non-finite values (NaN or Inf)")


_DEFAULT_EMBEDDER: Optional[SentenceTransformerEmbedder] = None


def get_embedder(
    model_name: str = SentenceTransformerEmbedder.DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
) -> SentenceTransformerEmbedder:
    """Returns a shared, cached SentenceTransformerEmbedder to avoid reloading models."""
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None or _DEFAULT_EMBEDDER.model_name != model_name:
        _DEFAULT_EMBEDDER = SentenceTransformerEmbedder(
            model_name=model_name, device=device
        )
    return _DEFAULT_EMBEDDER


def reset_embedder() -> None:
    """Resets the shared embedder instance, useful for testing and clean teardown."""
    global _DEFAULT_EMBEDDER
    _DEFAULT_EMBEDDER = None


def embed_chunks(
    chunks: List[TextChunk],
    embedder: Optional[BaseEmbedder] = None,
    batch_size: Optional[int] = None,
) -> EmbeddingResult:
    """Convenience function to embed a list of TextChunk objects.

    Reuses the shared embedder by default to avoid reloading the model repeatedly.
    """
    if embedder is None:
        embedder = get_embedder()
    return embedder.embed_chunks(chunks, batch_size=batch_size)
