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
from app.indexing.bm25_index import BM25Index
from app.indexing.multimodal_faiss_index import MultimodalFAISSIndex
from app.indexing.multimodal_models import MultimodalSearchResult
from app.multimodal.collection import MultimodalDocument
from app.multimodal.embedding import BaseVisualEmbedder
from app.retrieval.multimodal_assembler import MultimodalContextAssembler
from app.retrieval.multimodal_models import MultimodalRetrievedContext
from app.retrieval.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion


class MultimodalRetrievalPipeline:
    """Orchestrates similarity search across modality-specific FAISS indices and BM25 sparse index.

    Enforces strict modality isolation without vector concatenation or projection layers:
    - Text queries (strings or 384-d vectors) search ONLY text-like elements.
    - Visual queries (images or 512-d vectors) search ONLY visual elements.
    - Sparse queries search ONLY text-like elements via BM25 (IMAGE excluded).
    """

    def __init__(
        self,
        index: MultimodalFAISSIndex,
        text_embedder: Optional[BaseEmbedder] = None,
        visual_embedder: Optional[BaseVisualEmbedder] = None,
        default_top_k: int = 5,
        assembler: Optional[MultimodalContextAssembler] = None,
        bm25_index: Optional[BM25Index] = None,
    ) -> None:
        if not isinstance(index, MultimodalFAISSIndex):
            raise TypeError(
                f"index must be an instance of MultimodalFAISSIndex, got {type(index).__name__}"
            )

        if default_top_k <= 0:
            raise ValueError(
                f"default_top_k must be a positive integer, got {default_top_k}"
            )

        if bm25_index is not None and not isinstance(bm25_index, BM25Index):
            raise TypeError(
                f"bm25_index must be an instance of BM25Index, got {type(bm25_index).__name__}"
            )

        self.index = index
        self._text_embedder = text_embedder
        self._visual_embedder = visual_embedder
        self.default_top_k = default_top_k
        self.assembler = assembler or MultimodalContextAssembler()
        self.bm25_index = bm25_index

    @property
    def has_bm25(self) -> bool:
        """True if BM25 index is configured for this pipeline."""
        return self.bm25_index is not None

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

    def retrieve_sparse(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[MultimodalSearchResult]:
        """Retrieves top-k relevant text-like elements matching a query string using BM25.

        Operates strictly on text-like elements (TEXT, CAPTION, TABLE, EQUATION).
        IMAGE elements are never returned.

        Args:
            query: Query text string.
            top_k: Optional limit on returned results (defaults to default_top_k).

        Returns:
            List of MultimodalSearchResult instances ranked by BM25 score descending.

        Raises:
            RuntimeError: If BM25Index is not configured on this pipeline.
            ValueError: If top_k <= 0.
            TypeError: If query is not a string.
        """
        if self.bm25_index is None:
            raise RuntimeError(
                "BM25 index is not configured for this pipeline. Provide a BM25Index to enable sparse retrieval."
            )

        effective_k = self.default_top_k if top_k is None else top_k
        if effective_k <= 0:
            raise ValueError(
                f"top_k must be a positive integer, got {effective_k}"
            )

        if not isinstance(query, str):
            raise TypeError(
                f"query must be a string, got {type(query).__name__}"
            )

        if not query.strip():
            return []

        return self.bm25_index.search(query=query, top_k=effective_k)

    def hybrid_retrieve_text(
        self,
        query: str,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        top_k: Optional[int] = None,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> List[MultimodalSearchResult]:
        """Performs hybrid text retrieval combining dense 384-d FAISS and sparse BM25 via RRF.

        Operates strictly on text-like elements (TEXT, CAPTION, TABLE, EQUATION).
        IMAGE elements are never returned in hybrid text results.

        Args:
            query: Query text string.
            dense_top_k: Optional limit on dense retrieval candidates (defaults to top_k or default_top_k).
            sparse_top_k: Optional limit on sparse retrieval candidates (defaults to top_k or default_top_k).
            top_k: Optional limit on final fused results (defaults to default_top_k).
            rrf_k: Smoothing constant for Reciprocal Rank Fusion (defaults to 60).

        Returns:
            List of MultimodalSearchResult ranked by RRF score descending.

        Raises:
            RuntimeError: If BM25Index is not configured on this pipeline.
            TypeError: If query is not a string, or top_k / rrf_k have invalid types.
            ValueError: If top_k <= 0, dense_top_k <= 0, sparse_top_k <= 0, or rrf_k <= 0.
        """
        if self.bm25_index is None:
            raise RuntimeError(
                "BM25 index is not configured for this pipeline. Provide a BM25Index to enable hybrid retrieval."
            )

        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")

        effective_top_k = self.default_top_k if top_k is None else top_k
        if effective_top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {effective_top_k}")

        effective_dense_k = effective_top_k if dense_top_k is None else dense_top_k
        if effective_dense_k <= 0:
            raise ValueError(f"dense_top_k must be a positive integer, got {effective_dense_k}")

        effective_sparse_k = effective_top_k if sparse_top_k is None else sparse_top_k
        if effective_sparse_k <= 0:
            raise ValueError(f"sparse_top_k must be a positive integer, got {effective_sparse_k}")

        if not query.strip():
            return []

        # 1. Dense retrieval (384-d FAISS)
        dense_results = self.retrieve_text(query=query, top_k=effective_dense_k)

        # 2. Sparse retrieval (BM25)
        sparse_results = self.retrieve_sparse(query=query, top_k=effective_sparse_k)

        # 3. Reciprocal Rank Fusion
        fused_results = reciprocal_rank_fusion(
            ranked_lists=[dense_results, sparse_results],
            k=rrf_k,
            top_k=effective_top_k,
        )

        return fused_results

    def retrieve(
        self,
        query: Union[str, np.ndarray, bytes, Any],
        modality: str = "text",
        top_k: Optional[int] = None,
    ) -> List[MultimodalSearchResult]:
        """Dispatches retrieval to the specified modality index.

        Args:
            query: Query text, image, or pre-computed vector.
            modality: Target modality stream ('text', 'visual', 'sparse', or 'hybrid').
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
        if mod in ("sparse", "bm25"):
            return self.retrieve_sparse(query=query, top_k=top_k)
        if mod in ("hybrid", "hybrid_text", "dense_sparse"):
            if not isinstance(query, str):
                raise TypeError(f"Hybrid retrieval query must be a string, got {type(query).__name__}")
            return self.hybrid_retrieve_text(query=query, top_k=top_k)

        raise ValueError(
            f"Unsupported modality '{modality}'. Supported modalities are: 'text', 'visual', 'sparse', 'hybrid'"
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
            modality: Target modality stream ('text', 'visual', 'sparse', or 'hybrid'). Defaults to 'text'.
            top_k: Optional limit on returned results (defaults to default_top_k).
            document: Optional MultimodalDocument used for resolving inter-element relationships
                (such as image <-> caption links) during context assembly.

        Returns:
            MultimodalRetrievedContext containing formatted context, structured citations,
            and original MultimodalSearchResult objects.
        """
        search_results = self.retrieve(query=query, modality=modality, top_k=top_k)
        return self.assembler.assemble(search_results=search_results, document=document)

    def hybrid_retrieve_and_assemble(
        self,
        query: str,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        top_k: Optional[int] = None,
        rrf_k: int = DEFAULT_RRF_K,
        document: Optional[MultimodalDocument] = None,
    ) -> MultimodalRetrievedContext:
        """Retrieves multimodal elements via hybrid search and formats them into a prompt-ready context.

        Args:
            query: Query text string.
            dense_top_k: Optional limit on dense candidates.
            sparse_top_k: Optional limit on sparse candidates.
            top_k: Optional limit on final results.
            rrf_k: Smoothing constant for RRF.
            document: Optional MultimodalDocument for caption relationship resolution.

        Returns:
            MultimodalRetrievedContext with citations, formatted context, and fused search results.
        """
        search_results = self.hybrid_retrieve_text(
            query=query,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            top_k=top_k,
            rrf_k=rrf_k,
        )
        return self.assembler.assemble(search_results=search_results, document=document)

    # Convenience aliases
    search_text = retrieve_text
    search_visual = retrieve_visual
    search_sparse = retrieve_sparse
    search_hybrid_text = hybrid_retrieve_text
    retrieve_hybrid_text = hybrid_retrieve_text


def build_multimodal_pipeline(
    index: MultimodalFAISSIndex,
    text_embedder: Optional[BaseEmbedder] = None,
    visual_embedder: Optional[BaseVisualEmbedder] = None,
    default_top_k: int = 5,
    assembler: Optional[MultimodalContextAssembler] = None,
    bm25_index: Optional[BM25Index] = None,
) -> MultimodalRetrievalPipeline:
    """Convenience function to build a MultimodalRetrievalPipeline."""
    return MultimodalRetrievalPipeline(
        index=index,
        text_embedder=text_embedder,
        visual_embedder=visual_embedder,
        default_top_k=default_top_k,
        assembler=assembler,
        bm25_index=bm25_index,
    )


