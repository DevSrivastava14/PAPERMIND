from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from app.chunking.models import TextChunk


@dataclass
class EmbeddedChunk:
    """Represents an individual text chunk paired with its embedding vector and metadata."""

    chunk_id: str
    vector: np.ndarray
    metadata: Dict[str, Any]
    text: str


@dataclass
class EmbeddingResult:
    """Holds the batch embedding vectors and corresponding chunk metadata.

    Attributes:
        vectors: 2D NumPy array of shape (N, dimension) and dtype float32, ready for FAISS.
        chunk_ids: List of chunk IDs corresponding row-by-row to vectors.
        metadata: List of metadata dictionaries corresponding row-by-row to vectors.
        chunks: List of original TextChunk objects.
    """

    vectors: np.ndarray
    chunk_ids: List[str]
    metadata: List[Dict[str, Any]]
    chunks: List[TextChunk]

    def __len__(self) -> int:
        return len(self.chunks)

    def get_vector_by_id(self, chunk_id: str) -> Optional[np.ndarray]:
        """Returns the 1D embedding vector for a given chunk_id, or None if not found."""
        try:
            idx = self.chunk_ids.index(chunk_id)
            return self.vectors[idx]
        except ValueError:
            return None

    def to_embedded_chunks(self) -> List[EmbeddedChunk]:
        """Returns a list of EmbeddedChunk pairs linking each chunk with its vector."""
        return [
            EmbeddedChunk(
                chunk_id=self.chunk_ids[i],
                vector=self.vectors[i],
                metadata=self.metadata[i],
                text=self.chunks[i].text,
            )
            for i in range(len(self.chunks))
        ]
