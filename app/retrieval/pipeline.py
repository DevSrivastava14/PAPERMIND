from typing import List, Optional

from app.embeddings.base import BaseEmbedder
from app.indexing.faiss_index import FAISSIndex
from app.indexing.models import SearchResult
from app.retrieval.assembler import CitationContextAssembler
from app.retrieval.models import RetrievedContext


class RetrievalPipeline:
    """Orchestrates similarity search and context assembly across indexed research papers."""

    def __init__(
        self,
        index: FAISSIndex,
        embedder: Optional[BaseEmbedder] = None,
        default_top_k: int = 5,
        assembler: Optional[CitationContextAssembler] = None,
    ) -> None:
        if not isinstance(index, FAISSIndex):
            raise TypeError("index must be an instance of FAISSIndex")

        if default_top_k <= 0:
            raise ValueError(
                f"default_top_k must be a positive integer, got {default_top_k}"
            )

        self.index = index
        self.embedder = embedder
        self.default_top_k = default_top_k
        self.assembler = assembler or CitationContextAssembler()

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """Retrieves top-k relevant search results matching the query text.

        Supports a per-call override for top_k, falling back to default_top_k if not provided.
        """
        effective_k = self.default_top_k if top_k is None else top_k
        if effective_k <= 0:
            raise ValueError(
                f"top_k must be a positive integer, got {effective_k}"
            )

        if not isinstance(query, str) or not query.strip():
            return []

        return self.index.search(
            query=query, top_k=effective_k, embedder=self.embedder
        )

    def retrieve_and_assemble(
        self, query: str, top_k: Optional[int] = None
    ) -> RetrievedContext:
        """Retrieves relevant chunks and formats them into a prompt-ready RetrievedContext."""
        search_results = self.retrieve(query=query, top_k=top_k)
        return self.assembler.assemble(search_results)


def build_pipeline(
    index: FAISSIndex,
    embedder: Optional[BaseEmbedder] = None,
    default_top_k: int = 5,
) -> RetrievalPipeline:
    """Convenience function to build a RetrievalPipeline."""
    return RetrievalPipeline(
        index=index,
        embedder=embedder,
        default_top_k=default_top_k,
    )
