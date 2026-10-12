from unittest.mock import MagicMock
import numpy as np
import pytest

from app.chunking.models import TextChunk
from app.embeddings.base import BaseEmbedder
from app.embeddings.models import EmbeddingResult
from app.indexing.faiss_index import FAISSIndex, build_index
from app.indexing.models import SearchResult


def _create_sample_chunks(n: int = 3) -> list[TextChunk]:
    """Helper to create sample TextChunk objects."""
    return [
        TextChunk(
            chunk_id=f"paper.pdf#chunk_{i}",
            text=f"Text content of chunk {i}",
            source_filename="paper.pdf",
            chunk_index=i,
            page_numbers=[i + 1],
        )
        for i in range(n)
    ]


def _create_sample_embedding_result(
    chunks: list[TextChunk],
    vectors: np.ndarray,
) -> EmbeddingResult:
    """Helper to package chunks and vectors into an EmbeddingResult."""
    return EmbeddingResult(
        vectors=vectors,
        chunk_ids=[c.chunk_id for c in chunks],
        metadata=[
            {
                "chunk_id": c.chunk_id,
                "source_filename": c.source_filename,
                "chunk_index": c.chunk_index,
                "page_numbers": list(c.page_numbers),
            }
            for c in chunks
        ],
        chunks=list(chunks),
    )


class TestFAISSIndexing:
    """Unit tests for FAISS indexing, cosine similarity search, and provenance mapping."""

    def test_empty_index_search(self) -> None:
        """Verifies that searching an empty index returns an empty list without crashing."""
        index = FAISSIndex(dimension=384)
        assert index.total_chunks == 0
        assert len(index) == 0

        # Search by text query
        results = index.search("sample query", top_k=5)
        assert results == []

        # Search by vector
        query_vec = np.ones((384,), dtype=np.float32)
        vec_results = index.search_by_vector(query_vec, top_k=5)
        assert vec_results == []

    def test_empty_query_search(self) -> None:
        """Verifies that empty or whitespace text queries return an empty list."""
        chunks = _create_sample_chunks(2)
        vectors = np.ones((2, 4), dtype=np.float32)
        emb_res = _create_sample_embedding_result(chunks, vectors)

        index = FAISSIndex.from_embedding_result(emb_res)
        assert index.search("") == []
        assert index.search("   ") == []

    def test_invalid_top_k(self) -> None:
        """Verifies that top_k <= 0 raises ValueError."""
        index = FAISSIndex(dimension=4)
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            index.search("test", top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            index.search_by_vector(np.ones((4,), dtype=np.float32), top_k=-1)

    def test_add_empty_embedding_result(self) -> None:
        """Verifies that adding an empty EmbeddingResult does not alter index."""
        index = FAISSIndex(dimension=4)
        empty_res = EmbeddingResult(
            vectors=np.empty((0, 4), dtype=np.float32),
            chunk_ids=[],
            metadata=[],
            chunks=[],
        )
        index.add_embedding_result(empty_res)
        assert index.total_chunks == 0
        assert len(index) == 0

    def test_correct_ranking_and_cosine_normalization(self) -> None:
        """Verifies L2 normalization and cosine similarity ranking with known angles."""
        # 3 directional chunks in 3D:
        # chunk 0: along X axis (angle 0° to query)
        # chunk 1: 45° between X and Y
        # chunk 2: along Y axis (angle 90° to query)
        chunks = _create_sample_chunks(3)
        vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        emb_res = _create_sample_embedding_result(chunks, vectors)
        index = FAISSIndex.from_embedding_result(emb_res)

        # Query along X axis with arbitrary magnitude to verify query normalization
        query_vector = np.array([5.0, 0.0, 0.0], dtype=np.float32)
        results = index.search_by_vector(query_vector, top_k=3)

        assert len(results) == 3

        # Rank 0 should be chunk 0 (cos = 1.0)
        assert results[0].chunk_id == "paper.pdf#chunk_0"
        assert pytest.approx(results[0].score, abs=1e-5) == 1.0

        # Rank 1 should be chunk 1 (cos = 1/sqrt(2) ≈ 0.7071)
        assert results[1].chunk_id == "paper.pdf#chunk_1"
        assert pytest.approx(results[1].score, abs=1e-4) == 0.7071

        # Rank 2 should be chunk 2 (cos = 0.0)
        assert results[2].chunk_id == "paper.pdf#chunk_2"
        assert pytest.approx(results[2].score, abs=1e-5) == 0.0

    def test_chunk_provenance_and_id_mapping(self) -> None:
        """Verifies that search results preserve chunk provenance and ID lookups."""
        chunks = [
            TextChunk(
                chunk_id="test_doc.pdf#chunk_0",
                text="Attention is all you need.",
                source_filename="test_doc.pdf",
                chunk_index=0,
                page_numbers=[1, 2],
            ),
        ]
        vectors = np.array([[1.0, 2.0]], dtype=np.float32)
        emb_res = _create_sample_embedding_result(chunks, vectors)

        index = build_index(emb_res)
        assert index.total_chunks == 1

        # Direct lookups
        assert index.get_chunk_by_id(0) == chunks[0]
        assert index.get_chunk_by_chunk_id("test_doc.pdf#chunk_0") == chunks[0]
        assert index.get_chunk_by_id(999) is None
        assert index.get_chunk_by_chunk_id("missing") is None

        # Vector search result structure
        results = index.search_by_vector(np.array([1.0, 2.0], dtype=np.float32), top_k=1)
        assert len(results) == 1
        res = results[0]
        assert isinstance(res, SearchResult)
        assert res.chunk_id == "test_doc.pdf#chunk_0"
        assert res.source_filename == "test_doc.pdf"
        assert res.page_numbers == [1, 2]
        assert res.chunk == chunks[0]
        assert res.chunk.text == "Attention is all you need."
        assert pytest.approx(res.score, abs=1e-5) == 1.0

    def test_fewer_than_k_results(self) -> None:
        """Verifies that requesting top_k larger than index size never returns invalid indices."""
        chunks = _create_sample_chunks(2)
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        emb_res = _create_sample_embedding_result(chunks, vectors)

        index = FAISSIndex.from_embedding_result(emb_res)
        # Request top_k=10 on an index with only 2 chunks
        results = index.search_by_vector(np.array([1.0, 0.0], dtype=np.float32), top_k=10)

        # Must return exactly 2 items, discarding FAISS's internal -1 fillers
        assert len(results) == 2
        assert results[0].chunk_id == "paper.pdf#chunk_0"
        assert results[1].chunk_id == "paper.pdf#chunk_1"

    def test_text_search_with_mock_embedder(self) -> None:
        """Verifies search(query) delegates text embedding to the embedder."""
        chunks = _create_sample_chunks(2)
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        emb_res = _create_sample_embedding_result(chunks, vectors)
        index = FAISSIndex.from_embedding_result(emb_res)

        mock_embedder = MagicMock(spec=BaseEmbedder)
        mock_embedder.embed_texts.return_value = np.array([[1.0, 0.0]], dtype=np.float32)

        results = index.search("query text", top_k=2, embedder=mock_embedder)

        mock_embedder.embed_texts.assert_called_once_with(["query text"])
        assert len(results) == 2
        assert results[0].chunk_id == "paper.pdf#chunk_0"

    def test_invalid_numerical_inputs(self) -> None:
        """Verifies validation on non-finite values, dimension mismatches, and invalid types."""
        chunks = _create_sample_chunks(2)

        # NaN in vectors
        nan_vectors = np.array([[np.nan, 1.0], [1.0, 1.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite values"):
            emb_res = _create_sample_embedding_result(chunks, nan_vectors)
            FAISSIndex.from_embedding_result(emb_res)

        # Inf in vectors
        inf_vectors = np.array([[np.inf, 1.0], [1.0, 1.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite values"):
            emb_res = _create_sample_embedding_result(chunks, inf_vectors)
            FAISSIndex.from_embedding_result(emb_res)

        # Dimension mismatch
        valid_vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        emb_res = _create_sample_embedding_result(chunks, valid_vectors)
        index = FAISSIndex(dimension=4)  # expects 4, provided 2
        with pytest.raises(ValueError, match="dimension 2 does not match index dimension 4"):
            index.add_embedding_result(emb_res)

        # Query vector with NaN
        index2 = FAISSIndex.from_embedding_result(emb_res)
        with pytest.raises(ValueError, match="query_vector contains non-finite"):
            index2.search_by_vector(np.array([np.nan, 0.0], dtype=np.float32))

        # Query vector with dimension mismatch
        with pytest.raises(ValueError, match="query_vector dimension"):
            index2.search_by_vector(np.array([1.0, 0.0, 0.0], dtype=np.float32))

        # Invalid dimension at init
        with pytest.raises(ValueError, match="dimension must be positive"):
            FAISSIndex(dimension=0)
