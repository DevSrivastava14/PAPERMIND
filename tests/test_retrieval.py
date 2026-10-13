from unittest.mock import MagicMock
import numpy as np
import pytest

from app.chunking.models import TextChunk
from app.embeddings.base import BaseEmbedder
from app.embeddings.models import EmbeddingResult
from app.indexing.faiss_index import FAISSIndex
from app.indexing.models import SearchResult
from app.retrieval import (
    Citation,
    CitationContextAssembler,
    RetrievedContext,
    RetrievalPipeline,
    assemble_context,
    build_pipeline,
    format_page_reference,
)


def _create_search_result(
    chunk_id: str,
    text: str,
    source_filename: str,
    page_numbers: list[int],
    score: float,
) -> SearchResult:
    """Helper to create synthetic SearchResult objects."""
    chunk = TextChunk(
        chunk_id=chunk_id,
        text=text,
        source_filename=source_filename,
        chunk_index=0,
        page_numbers=page_numbers,
    )
    return SearchResult(
        chunk=chunk,
        score=score,
        chunk_id=chunk_id,
        source_filename=source_filename,
        page_numbers=page_numbers,
    )


class TestCitationAndAssembler:
    """Tests for Citation data model and CitationContextAssembler formatting."""

    def test_page_reference_formatting(self) -> None:
        """Verifies clean, non-destructive page reference formatting."""
        assert format_page_reference([]) == "Page unknown"
        assert format_page_reference([1]) == "Page 1"
        assert format_page_reference([3, 1, 2]) == "Pages 1-3"
        assert format_page_reference([1, 4]) == "Pages 1, 4"
        assert format_page_reference([2, 5, 3]) == "Pages 2, 3, 5"

    def test_assembler_normal_results(self) -> None:
        """Verifies context formatting and structured citation alignment."""
        results = [
            _create_search_result(
                chunk_id="paper.pdf#chunk_0",
                text="Attention mechanism allows modeling of dependencies without regard to distance.",
                source_filename="paper.pdf",
                page_numbers=[1, 2],
                score=0.92,
            ),
            _create_search_result(
                chunk_id="paper.pdf#chunk_1",
                text="The Transformer achieves 28.4 BLEU on the WMT 2014 English-to-German translation task.",
                source_filename="paper.pdf",
                page_numbers=[5],
                score=0.85,
            ),
        ]

        context = assemble_context(results)

        assert isinstance(context, RetrievedContext)
        assert len(context) == 2
        assert len(context.citations) == 2
        assert len(context.search_results) == 2

        # Check Citation 1
        c1 = context.citations[0]
        assert c1.citation_label == "[1]"
        assert c1.source_filename == "paper.pdf"
        assert c1.page_numbers == [1, 2]
        assert c1.chunk_id == "paper.pdf#chunk_0"
        assert c1.score == 0.92

        # Check Citation 2
        c2 = context.citations[1]
        assert c2.citation_label == "[2]"
        assert c2.source_filename == "paper.pdf"
        assert c2.page_numbers == [5]
        assert c2.chunk_id == "paper.pdf#chunk_1"
        assert c2.score == 0.85

        # Check formatted context contains markers and source/page references
        assert "[1] Source: paper.pdf (Pages 1-2)" in context.formatted_context
        assert (
            "Attention mechanism allows modeling of dependencies"
            in context.formatted_context
        )
        assert "[2] Source: paper.pdf (Page 5)" in context.formatted_context
        assert (
            "The Transformer achieves 28.4 BLEU" in context.formatted_context
        )

        # Evidence blocks are cleanly separated
        blocks = context.formatted_context.split("\n\n")
        assert len(blocks) == 2

    def test_assembler_empty_results(self) -> None:
        """Verifies assembler gracefully handles empty search results."""
        assembler = CitationContextAssembler()
        context = assembler.assemble([])

        assert isinstance(context, RetrievedContext)
        assert context.formatted_context == ""
        assert context.citations == []
        assert context.search_results == []
        assert len(context) == 0


class TestRetrievalPipeline:
    """Tests for RetrievalPipeline orchestration, overrides, and validation."""

    def test_init_validation(self) -> None:
        """Verifies validation on index type and default_top_k."""
        mock_index = MagicMock(spec=FAISSIndex)
        # Should succeed with mock FAISSIndex
        pipeline = RetrievalPipeline(index=mock_index, default_top_k=3)
        assert pipeline.default_top_k == 3

        # Invalid index type
        with pytest.raises(TypeError, match="must be an instance of FAISSIndex"):
            RetrievalPipeline(index="not_an_index")  # type: ignore

        # Invalid default_top_k
        with pytest.raises(
            ValueError, match="default_top_k must be a positive integer"
        ):
            RetrievalPipeline(index=mock_index, default_top_k=0)

        with pytest.raises(
            ValueError, match="default_top_k must be a positive integer"
        ):
            RetrievalPipeline(index=mock_index, default_top_k=-2)

    def test_retrieve_default_top_k_and_override(self) -> None:
        """Verifies retrieve delegates to FAISSIndex with default or overridden top_k."""
        mock_index = MagicMock(spec=FAISSIndex)
        mock_index.search.return_value = [
            _create_search_result(
                chunk_id="doc.pdf#chunk_0",
                text="Sample chunk",
                source_filename="doc.pdf",
                page_numbers=[1],
                score=0.99,
            )
        ]
        mock_embedder = MagicMock(spec=BaseEmbedder)

        pipeline = build_pipeline(
            index=mock_index, embedder=mock_embedder, default_top_k=4
        )

        # Call with default top_k
        res1 = pipeline.retrieve("transformer query")
        mock_index.search.assert_called_once_with(
            query="transformer query", top_k=4, embedder=mock_embedder
        )
        assert len(res1) == 1

        mock_index.search.reset_mock()

        # Call with top_k override
        res2 = pipeline.retrieve("transformer query", top_k=10)
        mock_index.search.assert_called_once_with(
            query="transformer query", top_k=10, embedder=mock_embedder
        )
        assert len(res2) == 1

    def test_invalid_top_k_override(self) -> None:
        """Verifies ValueError when providing non-positive top_k override."""
        mock_index = MagicMock(spec=FAISSIndex)
        pipeline = RetrievalPipeline(index=mock_index, default_top_k=5)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            pipeline.retrieve("query", top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            pipeline.retrieve("query", top_k=-1)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            pipeline.retrieve_and_assemble("query", top_k=0)

    def test_empty_and_whitespace_query(self) -> None:
        """Verifies empty or whitespace queries return empty results without searching index."""
        mock_index = MagicMock(spec=FAISSIndex)
        pipeline = RetrievalPipeline(index=mock_index)

        assert pipeline.retrieve("") == []
        assert pipeline.retrieve("   ") == []
        mock_index.search.assert_not_called()

        ctx_empty = pipeline.retrieve_and_assemble("")
        assert ctx_empty.formatted_context == ""
        assert ctx_empty.citations == []

    def test_query_to_context_flow(self) -> None:
        """Verifies end-to-end retrieve_and_assemble flow with FAISSIndex and mock embedder."""
        chunk1 = TextChunk(
            chunk_id="attention.pdf#chunk_0",
            text="The Transformer is based solely on attention mechanisms.",
            source_filename="attention.pdf",
            chunk_index=0,
            page_numbers=[1],
        )
        chunk2 = TextChunk(
            chunk_id="attention.pdf#chunk_1",
            text="Multi-head attention allows the model to jointly attend to information.",
            source_filename="attention.pdf",
            chunk_index=1,
            page_numbers=[3, 4],
        )
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        emb_res = EmbeddingResult(
            vectors=vectors,
            chunk_ids=[chunk1.chunk_id, chunk2.chunk_id],
            metadata=[
                {
                    "chunk_id": chunk1.chunk_id,
                    "source_filename": chunk1.source_filename,
                    "chunk_index": 0,
                    "page_numbers": [1],
                },
                {
                    "chunk_id": chunk2.chunk_id,
                    "source_filename": chunk2.source_filename,
                    "chunk_index": 1,
                    "page_numbers": [3, 4],
                },
            ],
            chunks=[chunk1, chunk2],
        )

        index = FAISSIndex.from_embedding_result(emb_res)

        mock_embedder = MagicMock(spec=BaseEmbedder)
        # Query directed along chunk1 direction
        mock_embedder.embed_texts.return_value = np.array(
            [[1.0, 0.0]], dtype=np.float32
        )

        pipeline = RetrievalPipeline(
            index=index, embedder=mock_embedder, default_top_k=2
        )

        context = pipeline.retrieve_and_assemble("what is attention mechanism?")

        assert len(context) == 2
        assert (
            context.citations[0].chunk_id == "attention.pdf#chunk_0"
        )
        assert context.citations[0].page_numbers == [1]
        assert context.citations[0].citation_label == "[1]"
        assert pytest.approx(context.citations[0].score, abs=1e-5) == 1.0

        assert (
            context.citations[1].chunk_id == "attention.pdf#chunk_1"
        )
        assert context.citations[1].page_numbers == [3, 4]
        assert context.citations[1].citation_label == "[2]"
        assert pytest.approx(context.citations[1].score, abs=1e-5) == 0.0

        assert "[1] Source: attention.pdf (Page 1)" in context.formatted_context
        assert (
            "[2] Source: attention.pdf (Pages 3-4)" in context.formatted_context
        )
