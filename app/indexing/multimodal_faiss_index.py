"""Modality-specific FAISS indexing for multimodal scientific paper elements.

This module provides separate, unconcatenated vector indices for text-like elements
(384 dimensions) and visual elements (512 dimensions) using FAISS IndexFlatIP with
L2-normalization for cosine similarity search.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import faiss
import numpy as np

from app.indexing.multimodal_models import MultimodalSearchResult
from app.multimodal.embedding import (
    BaseMultimodalEmbedder,
    BaseVisualEmbedder,
    MultimodalEmbeddedElement,
    MultimodalEmbeddingResult,
)
from app.multimodal.representation import ElementType, MultimodalElement


class ModalityFAISSIndex:
    """In-memory FAISS IndexFlatIP index for a single modality vector space.

    Enforces a single fixed dimensionality, L2-normalizes vectors prior to
    indexing and search for exact cosine similarity, and maps FAISS internal
    integer IDs back to the original MultimodalElement objects.
    """

    def __init__(self, dimension: int, name: str = "modality") -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be a positive integer, got {dimension}")

        self.dimension = int(dimension)
        self.name = name
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self._id_to_element: Dict[int, MultimodalElement] = {}
        self._element_id_to_internal_id: Dict[str, int] = {}
        self._total_elements: int = 0

    @property
    def total_elements(self) -> int:
        """Total number of elements currently indexed."""
        return self._total_elements

    def __len__(self) -> int:
        return self._total_elements

    def get_element_by_id(self, internal_id: int) -> Optional[MultimodalElement]:
        """Retrieves a MultimodalElement by its 0-based FAISS internal integer ID."""
        return self._id_to_element.get(internal_id)

    def get_element_by_element_id(self, element_id: str) -> Optional[MultimodalElement]:
        """Retrieves a MultimodalElement by its string element_id."""
        internal_id = self._element_id_to_internal_id.get(element_id)
        if internal_id is not None:
            return self._id_to_element.get(internal_id)
        return None

    def get_internal_id_by_element_id(self, element_id: str) -> Optional[int]:
        """Retrieves the 0-based FAISS internal integer ID for a string element_id."""
        return self._element_id_to_internal_id.get(element_id)

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """L2-normalizes vectors so inner product corresponds to cosine similarity."""
        if vectors.size == 0:
            return vectors.astype(np.float32, copy=False)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (vectors / norms).astype(np.float32)

    def _validate_vectors(self, vectors: np.ndarray, expected_count: int) -> None:
        """Validates that vectors are 2D float32 arrays with finite values matching dimension."""
        if not isinstance(vectors, np.ndarray):
            raise TypeError(
                f"vectors must be a NumPy ndarray, got {type(vectors).__name__}"
            )

        if vectors.ndim != 2:
            raise ValueError(
                f"vectors must be a 2D array, got {vectors.ndim}D array with shape {vectors.shape}"
            )

        if vectors.shape[0] != expected_count:
            raise ValueError(
                f"Vector count ({vectors.shape[0]}) does not match elements count ({expected_count})"
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension} "
                f"for {self.name} index"
            )

        if not np.all(np.isfinite(vectors)):
            raise ValueError("vectors contain non-finite values (NaN or Inf)")

    def add_elements(
        self,
        elements: Sequence[MultimodalElement],
        vectors: Optional[np.ndarray] = None,
    ) -> None:
        """Adds MultimodalElement instances and their embedding vectors to the index.

        If vectors is not provided, extracts embeddings from each element's .embedding attribute.
        L2-normalizes vectors prior to insertion so IndexFlatIP computes cosine similarity.
        """
        if not elements:
            return

        for idx, el in enumerate(elements):
            if not isinstance(el, MultimodalElement):
                raise TypeError(
                    f"Element at index {idx} must be a MultimodalElement, got {type(el).__name__}"
                )

        if vectors is None:
            extracted = []
            for idx, el in enumerate(elements):
                if el.embedding is None:
                    raise ValueError(
                        f"Element '{el.element_id}' at index {idx} does not have an embedding"
                    )
                vec = np.asarray(el.embedding, dtype=np.float32)
                if vec.ndim != 1:
                    raise ValueError(
                        f"Embedding for element '{el.element_id}' must be 1D, got {vec.ndim}D"
                    )
                extracted.append(vec)
            vectors = np.vstack(extracted)

        self._validate_vectors(vectors, expected_count=len(elements))

        normalized_vectors = self._normalize_vectors(vectors)
        normalized_vectors = np.ascontiguousarray(normalized_vectors, dtype=np.float32)

        start_id = self._total_elements
        self._index.add(normalized_vectors)

        for i, el in enumerate(elements):
            internal_id = start_id + i
            self._id_to_element[internal_id] = el
            self._element_id_to_internal_id[el.element_id] = internal_id

        self._total_elements += len(elements)

    def search_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[MultimodalSearchResult]:
        """Performs cosine similarity search using a pre-computed query vector."""
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if self._total_elements == 0:
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
            raise ValueError("query_vector contains non-finite values (NaN or Inf)")

        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"query_vector dimension ({query_vector.shape[1]}) does not match "
                f"index dimension ({self.dimension}) for {self.name} index"
            )

        normalized_query = self._normalize_vectors(query_vector)
        normalized_query = np.ascontiguousarray(normalized_query, dtype=np.float32)

        k = min(top_k, self._total_elements)
        scores, indices = self._index.search(normalized_query, k)

        results: List[MultimodalSearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx not in self._id_to_element:
                continue

            element = self._id_to_element[idx]
            results.append(
                MultimodalSearchResult(
                    element=element,
                    score=float(score),
                    element_id=element.element_id,
                    element_type=element.element_type,
                    source_filename=element.source_filename,
                    page_number=element.page_number,
                    related_element_ids=list(element.related_element_ids),
                )
            )

        return results


class MultimodalFAISSIndex:
    """Composite multimodal index managing separate vector indices for text and visual spaces.

    - Text-like elements (text, captions, tables, equations) are indexed in a 384-dimensional space.
    - Visual elements (images) are indexed in a 512-dimensional space.
    - Text and visual embeddings remain in separate vector spaces: no vector concatenation,
      no projection layer, and no single combined FAISS index.
    """

    DEFAULT_TEXT_DIMENSION: int = 384
    DEFAULT_VISUAL_DIMENSION: int = 512

    TEXT_LIKE_TYPES = {
        ElementType.TEXT,
        ElementType.CAPTION,
        ElementType.TABLE,
        ElementType.EQUATION,
    }
    VISUAL_TYPES = {
        ElementType.IMAGE,
    }

    def __init__(
        self,
        text_dimension: int = DEFAULT_TEXT_DIMENSION,
        visual_dimension: int = DEFAULT_VISUAL_DIMENSION,
    ) -> None:
        self.text_dimension = int(text_dimension)
        self.visual_dimension = int(visual_dimension)

        self.text_index = ModalityFAISSIndex(
            dimension=self.text_dimension,
            name="text",
        )
        self.visual_index = ModalityFAISSIndex(
            dimension=self.visual_dimension,
            name="visual",
        )

    @property
    def total_text_elements(self) -> int:
        """Total number of elements in the text index."""
        return self.text_index.total_elements

    @property
    def total_visual_elements(self) -> int:
        """Total number of elements in the visual index."""
        return self.visual_index.total_elements

    @property
    def total_elements(self) -> int:
        """Total number of elements across both modality indices."""
        return self.total_text_elements + self.total_visual_elements

    def __len__(self) -> int:
        return self.total_elements

    def add_embedding_result(
        self,
        embedding_result: MultimodalEmbeddingResult,
    ) -> None:
        """Routes embedded elements into their modality-specific indices.

        Text, caption, table, and equation elements are routed to the 384-d text index.
        Image elements are routed to the 512-d visual index.
        """
        if not isinstance(embedding_result, MultimodalEmbeddingResult):
            raise TypeError(
                f"embedding_result must be a MultimodalEmbeddingResult, got {type(embedding_result).__name__}"
            )

        if len(embedding_result) == 0:
            return

        text_elements: List[MultimodalElement] = []
        text_vectors: List[np.ndarray] = []

        visual_elements: List[MultimodalElement] = []
        visual_vectors: List[np.ndarray] = []

        for ee in embedding_result.embedded_elements:
            el = ee.element
            vec = ee.vector

            if el.element_type in self.TEXT_LIKE_TYPES:
                text_elements.append(el)
                text_vectors.append(vec)
            elif el.element_type in self.VISUAL_TYPES:
                visual_elements.append(el)
                visual_vectors.append(vec)
            else:
                raise ValueError(
                    f"Unsupported element_type '{el.element_type}' for element '{el.element_id}'"
                )

        if text_elements:
            text_matrix = np.vstack(text_vectors).astype(np.float32)
            self.text_index.add_elements(text_elements, vectors=text_matrix)

        if visual_elements:
            visual_matrix = np.vstack(visual_vectors).astype(np.float32)
            self.visual_index.add_elements(visual_elements, vectors=visual_matrix)

    def add_document(
        self,
        document: Any,
        embedder: Optional[BaseMultimodalEmbedder] = None,
    ) -> None:
        """Adds all elements of a MultimodalDocument into the multimodal index.

        If the document is not yet fully embedded, computes embeddings using embedder.
        """
        if not hasattr(document, "elements"):
            raise TypeError("document must have an 'elements' attribute")

        if getattr(document, "is_fully_embedded", False):
            # Already embedded: partition directly
            text_elements: List[MultimodalElement] = []
            visual_elements: List[MultimodalElement] = []

            for el in document.elements:
                if el.embedding is None:
                    continue
                if el.element_type in self.TEXT_LIKE_TYPES:
                    text_elements.append(el)
                elif el.element_type in self.VISUAL_TYPES:
                    visual_elements.append(el)
                else:
                    raise ValueError(
                        f"Unsupported element_type '{el.element_type}' for element '{el.element_id}'"
                    )

            if text_elements:
                self.text_index.add_elements(text_elements)
            if visual_elements:
                self.visual_index.add_elements(visual_elements)
        else:
            # Generate embeddings and add result
            emb_res = document.embed(embedder=embedder)
            self.add_embedding_result(emb_res)

    def search_text_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[MultimodalSearchResult]:
        """Performs cosine similarity search against the text-like index using a vector."""
        return self.text_index.search_by_vector(query_vector=query_vector, top_k=top_k)

    def search_visual_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[MultimodalSearchResult]:
        """Performs cosine similarity search against the visual index using a vector."""
        return self.visual_index.search_by_vector(query_vector=query_vector, top_k=top_k)

    def search_text(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 5,
        embedder: Optional[Any] = None,
    ) -> List[MultimodalSearchResult]:
        """Searches the text index with a query string or pre-computed text vector.

        Text queries only search the text index.
        """
        if isinstance(query, np.ndarray):
            return self.search_text_by_vector(query, top_k=top_k)

        if not isinstance(query, str) or not query.strip():
            return []

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if self.total_text_elements == 0:
            return []

        # Resolve text embedder
        active_embedder = embedder
        if active_embedder is None:
            from app.embeddings.sentence_transformer_embedder import get_embedder

            active_embedder = get_embedder()
        elif hasattr(active_embedder, "text_embedder"):
            active_embedder = active_embedder.text_embedder

        query_vectors = active_embedder.embed_texts([query])
        return self.search_text_by_vector(query_vectors[0], top_k=top_k)

    def search_visual(
        self,
        query: Union[np.ndarray, bytes, str, Any],
        top_k: int = 5,
        embedder: Optional[Union[BaseVisualEmbedder, Any]] = None,
    ) -> List[MultimodalSearchResult]:
        """Searches the visual index with an image (bytes/path/PIL) or pre-computed visual vector.

        Visual queries only search the visual index.
        """
        if isinstance(query, np.ndarray):
            return self.search_visual_by_vector(query, top_k=top_k)

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if self.total_visual_elements == 0:
            return []

        # Resolve visual embedder
        active_embedder = embedder
        if active_embedder is None:
            from app.multimodal.embedding import get_multimodal_embedder

            active_embedder = get_multimodal_embedder().visual_embedder
        elif hasattr(active_embedder, "visual_embedder"):
            active_embedder = active_embedder.visual_embedder

        query_vectors = active_embedder.embed_images([query])
        return self.search_visual_by_vector(query_vectors[0], top_k=top_k)

    @classmethod
    def from_embedding_result(
        cls,
        embedding_result: MultimodalEmbeddingResult,
        text_dimension: int = DEFAULT_TEXT_DIMENSION,
        visual_dimension: int = DEFAULT_VISUAL_DIMENSION,
    ) -> "MultimodalFAISSIndex":
        """Builds a new MultimodalFAISSIndex directly from a MultimodalEmbeddingResult."""
        index = cls(text_dimension=text_dimension, visual_dimension=visual_dimension)
        index.add_embedding_result(embedding_result)
        return index

    @classmethod
    def from_document(
        cls,
        document: Any,
        embedder: Optional[BaseMultimodalEmbedder] = None,
        text_dimension: int = DEFAULT_TEXT_DIMENSION,
        visual_dimension: int = DEFAULT_VISUAL_DIMENSION,
    ) -> "MultimodalFAISSIndex":
        """Builds a new MultimodalFAISSIndex directly from a MultimodalDocument."""
        index = cls(text_dimension=text_dimension, visual_dimension=visual_dimension)
        index.add_document(document, embedder=embedder)
        return index


def build_multimodal_index(
    source: Union[MultimodalEmbeddingResult, Any],
    embedder: Optional[BaseMultimodalEmbedder] = None,
    text_dimension: int = MultimodalFAISSIndex.DEFAULT_TEXT_DIMENSION,
    visual_dimension: int = MultimodalFAISSIndex.DEFAULT_VISUAL_DIMENSION,
) -> MultimodalFAISSIndex:
    """Convenience function to build a MultimodalFAISSIndex from a document or embedding result."""
    if isinstance(source, MultimodalEmbeddingResult):
        return MultimodalFAISSIndex.from_embedding_result(
            embedding_result=source,
            text_dimension=text_dimension,
            visual_dimension=visual_dimension,
        )
    return MultimodalFAISSIndex.from_document(
        document=source,
        embedder=embedder,
        text_dimension=text_dimension,
        visual_dimension=visual_dimension,
    )
