"""Abstract base interface for candidate rerankers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from app.indexing.multimodal_models import MultimodalSearchResult


class BaseReranker(ABC):
    """Abstract base class defining the interface for candidate rerankers.

    Rerankers accept a search query and a sequence of candidate MultimodalSearchResult
    objects, re-score candidates using a relevance model (such as a cross-encoder),
    and return candidates sorted by relevance score descending while preserving
    all original provenance and element attributes.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[MultimodalSearchResult],
        top_k: Optional[int] = None,
    ) -> List[MultimodalSearchResult]:
        """Reranks candidate search results against a query string.

        Args:
            query: The search query text.
            candidates: Sequence of MultimodalSearchResult candidates to rerank.
            top_k: Optional maximum number of reranked candidates to return.
                If None, returns all scored candidates.

        Returns:
            List of MultimodalSearchResult instances sorted by relevance score descending.
        """
        pass

    def __call__(
        self,
        query: str,
        candidates: Sequence[MultimodalSearchResult],
        top_k: Optional[int] = None,
    ) -> List[MultimodalSearchResult]:
        """Convenience callable alias delegating directly to self.rerank."""
        return self.rerank(query=query, candidates=candidates, top_k=top_k)
