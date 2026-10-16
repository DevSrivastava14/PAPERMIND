"""Modality-aware multimodal retrieval pipeline for scientific paper elements.

This module provides a unified retrieval interface (`MultimodalRetrievalPipeline`)
operating over the modality-specific FAISS indices (`MultimodalFAISSIndex`).
Text and visual streams remain strictly isolated in their respective vector spaces:
- Text queries search ONLY the 384-dimensional text index.
- Visual queries search ONLY the 512-dimensional visual index.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np

from app.embeddings.base import BaseEmbedder
from app.indexing.multimodal_faiss_index import MultimodalFAISSIndex
from app.indexing.multimodal_models import MultimodalSearchResult
from app.multimodal.collection import MultimodalDocument
from app.multimodal.embedding import BaseVisualEmbedder
from app.retrieval.multimodal_assembler import MultimodalContextAssembler
from app.retrieval.multimodal_models import MultimodalRetrievedContext


class MultimodalRetrievalPipeline:
    """Orchestrates similarity search across modality-specific FAISS indices.

    Enforces strict modality isolation without vector concatenation or projection layers:
    - Text queries (strings or 384-d vectors) search ONLY text-like elements.
    - Visual queries (images or 512-d vectors) search ONLY visual elements.
    """

    def __init__(
        self,
        index: MultimodalFAISSIndex,
        text_embedder: Optional[BaseEmbedder] = None,
        visual_embedder: Optional[BaseVisualEmbedder] = None,
        default_top_k: int = 5,
        assembler: Optional[MultimodalContextAssembler] = None,
    ) -> None:
        if not isinstance(index, MultimodalFAISSIndex):
            raise TypeError(
                f"index must be an instance of MultimodalFAISSIndex, got {type(index).__name__}"
            )

        if default_top_k <= 0:
            raise ValueError(
                f"default_top_k must be a positive integer, got {default_top_k}"
            )

        self.index = index
        self._text_embedder = text_embedder
        self._visual_embedder = visual_embedder
        self.default_top_k = default_top_k
        self.assembler = assembler or MultimodalContextAssembler()

    @property
    def text_embedder(self) -> BaseEmbedder:
        """Returns the configured or default text embedder."""
        if self._text_embedder is None:
            from app.embeddings.sentence_transformer_embedder import get_embedder

            self._text_embedder = get_embedder()
        return self._text_embedder

    @property
    def visual_embedder(self) -> BaseVisualEmbedder:
        """Returns the configured or default visual embedder."""
        if self._visual_embedder is None:
            from app.multimodal.embedding import get_multimodal_embedder

            self._visual_embedder = get_multimodal_embedder().visual_embedder
        return self._visual_embedder

    def retrieve_text(
        self,
        query: Union[str, np.ndarray],
        top_k: Optional[int] = None,
    ) -> List[MultimodalSearchResult]:
        """Retrieves top-k relevant text-like elements matching a query string or 384-d vector.

        Searches ONLY the 384-dimensional text index.
        """
        effective_k = self.default_top_k if top_k is None else top_k
        if effective_k <= 0:
            raise ValueError(
                f"top_k must be a positive integer, got {effective_k}"
            )

        if isinstance(query, str):
            if not query.strip():
                return []
            return self.index.search_text(
                query=query,
                top_k=effective_k,
                embedder=self.text_embedder,
            )

        if isinstance(query, np.ndarray):
            return self.index.search_text_by_vector(
                query_vector=query,
                top_k=effective_k,
            )

        raise TypeError(
            f"query must be a string or NumPy ndarray, got {type(query).__name__}"
        )

    def retrieve_visual(
        self,
        query: Union[np.ndarray, bytes, str, Any],
        top_k: Optional[int] = None,
    ) -> List[MultimodalSearchResult]:
        """Retrieves top-k relevant visual elements matching an image or 512-d vector.

        Searches ONLY the 512-dimensional visual index.
        """
        effective_k = self.default_top_k if top_k is None else top_k
        if effective_k <= 0:
            raise ValueError(
                f"top_k must be a positive integer, got {effective_k}"
            )

        if isinstance(query, np.ndarray):
            return self.index.search_visual_by_vector(
                query_vector=query,
                top_k=effective_k,
            )

        return self.index.search_visual(
            query=query,
            top_k=effective_k,
            embedder=self.visual_embedder,
        )

    def retrieve(
        self,
        query: Union[str, np.ndarray, bytes, Any],
        modality: str = "text",
        top_k: Optional[int] = None,
    ) -> List[MultimodalSearchResult]:
        """Dispatches retrieval to the specified modality index.

        Args:
            query: Query text, image, or pre-computed vector.
            modality: Target modality stream ('text' or 'visual').
            top_k: Optional limit on returned results.
        """
        if not isinstance(modality, str):
            raise TypeError(
                f"modality must be a string, got {type(modality).__name__}"
            )

        mod = modality.strip().lower()
        if mod in ("text", "text_like", "caption", "table", "equation"):
            return self.retrieve_text(query=query, top_k=top_k)
        if mod in ("visual", "image", "figure"):
            return self.retrieve_visual(query=query, top_k=top_k)

        raise ValueError(
            f"Unsupported modality '{modality}'. Supported modalities are: 'text', 'visual'"
        )

    def retrieve_and_assemble(
        self,
        query: Union[str, np.ndarray, bytes, Any],
        modality: str = "text",
        top_k: Optional[int] = None,
        document: Optional[MultimodalDocument] = None,
    ) -> MultimodalRetrievedContext:
        """Retrieves relevant multimodal elements and formats them into a prompt-ready context.

        Args:
            query: Query text string, image payload, or pre-computed vector.
            modality: Target modality stream ('text' or 'visual'). Defaults to 'text'.
            top_k: Optional limit on returned results (defaults to default_top_k).
            document: Optional MultimodalDocument used for resolving inter-element relationships
                (such as image <-> caption links) during context assembly.

        Returns:
            MultimodalRetrievedContext containing formatted context, structured citations,
            and original MultimodalSearchResult objects.
        """
        search_results = self.retrieve(query=query, modality=modality, top_k=top_k)
        return self.assembler.assemble(search_results=search_results, document=document)

    # Convenience aliases
    search_text = retrieve_text
    search_visual = retrieve_visual


def build_multimodal_pipeline(
    index: MultimodalFAISSIndex,
    text_embedder: Optional[BaseEmbedder] = None,
    visual_embedder: Optional[BaseVisualEmbedder] = None,
    default_top_k: int = 5,
    assembler: Optional[MultimodalContextAssembler] = None,
) -> MultimodalRetrievalPipeline:
    """Convenience function to build a MultimodalRetrievalPipeline."""
    return MultimodalRetrievalPipeline(
        index=index,
        text_embedder=text_embedder,
        visual_embedder=visual_embedder,
        default_top_k=default_top_k,
        assembler=assembler,
    )

