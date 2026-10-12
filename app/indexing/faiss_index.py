from typing import Dict, List, Optional
import faiss
import numpy as np

from app.chunking.models import TextChunk
from app.embeddings.base import BaseEmbedder
from app.embeddings.models import EmbeddingResult
from app.indexing.models import SearchResult


class FAISSIndex:
    """In-memory FAISS IndexFlatIP index with L2 normalization for cosine similarity search."""

    def __init__(self, dimension: Optional[int] = None) -> None:
        if dimension is not None and dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

        self.dimension = dimension
        self._index: Optional[faiss.IndexFlatIP] = None
        self._id_to_chunk: Dict[int, TextChunk] = {}
        self._chunk_id_to_internal_id: Dict[str, int] = {}
        self._total_chunks: int = 0

        if dimension is not None:
            self._index = faiss.IndexFlatIP(dimension)

    @property
    def total_chunks(self) -> int:
        """Total number of text chunks currently indexed."""
        return self._total_chunks

    def __len__(self) -> int:
        return self._total_chunks

    def get_chunk_by_id(self, internal_id: int) -> Optional[TextChunk]:
        """Retrieves a TextChunk by its 0-based FAISS internal integer ID."""
        return self._id_to_chunk.get(internal_id)

    def get_chunk_by_chunk_id(self, chunk_id: str) -> Optional[TextChunk]:
        """Retrieves a TextChunk by its unique chunk string ID."""
        internal_id = self._chunk_id_to_internal_id.get(chunk_id)
        if internal_id is not None:
            return self._id_to_chunk.get(internal_id)
        return None

    def add_embedding_result(self, embedding_result: EmbeddingResult) -> None:
        """Validates and adds an EmbeddingResult to the FAISS index.

        L2-normalizes vectors prior to insertion so IndexFlatIP computes cosine similarity.
        """
        if not isinstance(embedding_result, EmbeddingResult):
            raise TypeError("embedding_result must be an instance of EmbeddingResult")

        if len(embedding_result.chunks) == 0:
            return

        vectors = embedding_result.vectors
        chunks = embedding_result.chunks
        chunk_ids = embedding_result.chunk_ids

        self._validate_input(vectors, chunks, chunk_ids)

        if self._index is None:
            self.dimension = vectors.shape[1]
            self._index = faiss.IndexFlatIP(self.dimension)

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}"
            )

        normalized_vectors = self._normalize_vectors(vectors)
        normalized_vectors = np.ascontiguousarray(
            normalized_vectors, dtype=np.float32
        )

        start_id = self._total_chunks
        self._index.add(normalized_vectors)

        for i, chunk in enumerate(chunks):
            internal_id = start_id + i
            self._id_to_chunk[internal_id] = chunk
            self._chunk_id_to_internal_id[chunk.chunk_id] = internal_id

        self._total_chunks += len(chunks)

    def search_by_vector(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[SearchResult]:
        """Performs cosine similarity search using a pre-computed query vector."""
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if self._total_chunks == 0 or self._index is None:
            return []

        if not isinstance(query_vector, np.ndarray):
            raise TypeError("query_vector must be a NumPy ndarray")

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        elif query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise ValueError(
                f"query_vector must have shape (dimension,) or (1, dimension), got shape {query_vector.shape}"
            )

        if not np.all(np.isfinite(query_vector)):
            raise ValueError(
                "query_vector contains non-finite values (NaN or Inf)"
            )

        if (
            self.dimension is not None
            and query_vector.shape[1] != self.dimension
        ):
            raise ValueError(
                f"query_vector dimension ({query_vector.shape[1]}) does not match index dimension ({self.dimension})"
            )

        normalized_query = self._normalize_vectors(query_vector)
        normalized_query = np.ascontiguousarray(
            normalized_query, dtype=np.float32
        )

        scores, indices = self._index.search(normalized_query, top_k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx not in self._id_to_chunk:
                continue

            chunk = self._id_to_chunk[idx]
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(score),
                    chunk_id=chunk.chunk_id,
                    source_filename=chunk.source_filename,
                    page_numbers=list(chunk.page_numbers),
                )
            )

        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        embedder: Optional[BaseEmbedder] = None,
    ) -> List[SearchResult]:
        """Embeds a text query and returns top-k cosine similarity search results."""
        if not isinstance(query, str) or not query.strip():
            return []

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if self._total_chunks == 0 or self._index is None:
            return []

        if embedder is None:
            from app.embeddings import get_embedder

            embedder = get_embedder()

        query_vectors = embedder.embed_texts([query])
        return self.search_by_vector(query_vectors[0], top_k=top_k)

    @classmethod
    def from_embedding_result(
        cls,
        embedding_result: EmbeddingResult,
        dimension: Optional[int] = None,
    ) -> "FAISSIndex":
        """Builds a new FAISSIndex directly from an EmbeddingResult."""
        dim = dimension
        if dim is None and embedding_result.vectors.size > 0:
            dim = embedding_result.vectors.shape[1]
        index = cls(dimension=dim)
        index.add_embedding_result(embedding_result)
        return index

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """L2-normalizes vectors so inner product corresponds to cosine similarity."""
        if vectors.size == 0:
            return vectors.astype(np.float32, copy=False)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (vectors / norms).astype(np.float32)

    def _validate_input(
        self,
        vectors: np.ndarray,
        chunks: List[TextChunk],
        chunk_ids: List[str],
    ) -> None:
        """Validates that vectors are 2D NumPy array with finite values matching chunk counts."""
        if not isinstance(vectors, np.ndarray):
            raise TypeError("vectors must be a NumPy ndarray")

        if vectors.ndim != 2:
            raise ValueError(
                f"vectors must be a 2D array, got {vectors.ndim}D array with shape {vectors.shape}"
            )

        if not np.all(np.isfinite(vectors)):
            raise ValueError("vectors contain non-finite values (NaN or Inf)")

        if len(chunks) != len(chunk_ids):
            raise ValueError(
                f"chunks count ({len(chunks)}) does not match chunk_ids count ({len(chunk_ids)})"
            )

        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"vectors count ({vectors.shape[0]}) does not match chunks count ({len(chunks)})"
            )


def build_index(
    embedding_result: EmbeddingResult,
    dimension: Optional[int] = None,
) -> FAISSIndex:
    """Convenience function to build a FAISSIndex from an EmbeddingResult."""
    return FAISSIndex.from_embedding_result(
        embedding_result=embedding_result, dimension=dimension
    )
