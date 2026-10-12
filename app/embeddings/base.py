from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from app.chunking.models import TextChunk
from app.embeddings.models import EmbeddingResult


class BaseEmbedder(ABC):
    """Abstract base class defining the interface for embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors produced by this model."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The identifier or name of the embedding model."""
        pass

    @abstractmethod
    def embed_texts(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Embeds a list of raw text strings into a 2D float32 NumPy array of shape (N, dimension)."""
        pass

    def embed_chunks(
        self, chunks: List[TextChunk], batch_size: Optional[int] = None
    ) -> EmbeddingResult:
        """Embeds a list of TextChunk objects and returns an EmbeddingResult preserving metadata."""
        if not chunks:
            return EmbeddingResult(
                vectors=np.empty((0, self.dimension), dtype=np.float32),
                chunk_ids=[],
                metadata=[],
                chunks=[],
            )

        texts = [chunk.text for chunk in chunks]
        vectors = self.embed_texts(texts, batch_size=batch_size)

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        metadata = [
            {
                "chunk_id": chunk.chunk_id,
                "source_filename": chunk.source_filename,
                "chunk_index": chunk.chunk_index,
                "page_numbers": list(chunk.page_numbers),
            }
            for chunk in chunks
        ]

        return EmbeddingResult(
            vectors=vectors,
            chunk_ids=chunk_ids,
            metadata=metadata,
            chunks=list(chunks),
        )
