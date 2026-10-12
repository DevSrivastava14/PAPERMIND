from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.chunking.models import TextChunk
from app.embeddings.base import BaseEmbedder
from app.embeddings.models import EmbeddedChunk, EmbeddingResult
from app.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    embed_chunks,
    get_embedder,
    reset_embedder,
)


def _create_sample_chunks(n: int = 3) -> list[TextChunk]:
    """Helper to generate sample TextChunk objects for testing."""
    return [
        TextChunk(
            chunk_id=f"sample_paper.pdf#chunk_{i}",
            text=f"This is chunk text content number {i}.",
            source_filename="sample_paper.pdf",
            chunk_index=i,
            page_numbers=[i + 1],
        )
        for i in range(n)
    ]


class TestEmbeddingPipeline:
    """Unit tests for the embedding pipeline, models, and batch processing."""

    def test_embed_empty_input(self) -> None:
        """Verifies that empty chunk and text lists return valid empty vectors without crashing."""
        mock_model = MagicMock()
        embedder = SentenceTransformerEmbedder(model=mock_model, dimension=384)

        # Test empty chunks
        chunk_result = embedder.embed_chunks([])
        assert isinstance(chunk_result, EmbeddingResult)
        assert len(chunk_result) == 0
        assert chunk_result.vectors.shape == (0, 384)
        assert chunk_result.vectors.dtype == np.float32
        assert chunk_result.chunk_ids == []
        assert chunk_result.metadata == []
        assert chunk_result.chunks == []

        # Test empty texts
        text_result = embedder.embed_texts([])
        assert isinstance(text_result, np.ndarray)
        assert text_result.shape == (0, 384)
        assert text_result.dtype == np.float32

        # Verify underlying model was never called
        mock_model.encode.assert_not_called()

    def test_output_dimensions_and_vector_count(self) -> None:
        """Verifies correct output array dimensions, dtype, and vector count."""
        chunks = _create_sample_chunks(3)
        mock_model = MagicMock()
        mock_vectors = np.ones((3, 384), dtype=np.float32)
        mock_model.encode.return_value = mock_vectors

        embedder = SentenceTransformerEmbedder(model=mock_model, dimension=384)
        result = embedder.embed_chunks(chunks)

        assert isinstance(result.vectors, np.ndarray)
        assert result.vectors.shape == (3, 384)
        assert result.vectors.dtype == np.float32
        assert len(result) == 3
        assert len(result.chunk_ids) == 3
        assert len(result.metadata) == 3
        assert len(result.chunks) == 3

    def test_chunk_to_vector_mapping_and_metadata_preservation(self) -> None:
        """Verifies 1:1 mapping between chunks, vectors, IDs, and separate metadata."""
        chunks = [
            TextChunk(
                chunk_id="doc.pdf#chunk_0",
                text="Text zero",
                source_filename="doc.pdf",
                chunk_index=0,
                page_numbers=[1],
            ),
            TextChunk(
                chunk_id="doc.pdf#chunk_1",
                text="Text one",
                source_filename="doc.pdf",
                chunk_index=1,
                page_numbers=[1, 2],
            ),
            TextChunk(
                chunk_id="doc.pdf#chunk_2",
                text="Text two",
                source_filename="doc.pdf",
                chunk_index=2,
                page_numbers=[2],
            ),
        ]

        # Distinct vectors for each chunk
        v0 = np.full((384,), 1.0, dtype=np.float32)
        v1 = np.full((384,), 2.0, dtype=np.float32)
        v2 = np.full((384,), 3.0, dtype=np.float32)

        mock_model = MagicMock()
        mock_model.encode.return_value = np.stack([v0, v1, v2])

        embedder = SentenceTransformerEmbedder(model=mock_model, dimension=384)
        result = embedder.embed_chunks(chunks)

        # Check preserved IDs
        assert result.chunk_ids == [
            "doc.pdf#chunk_0",
            "doc.pdf#chunk_1",
            "doc.pdf#chunk_2",
        ]

        # Check vector values match row-by-row
        assert np.allclose(result.vectors[0], 1.0)
        assert np.allclose(result.vectors[1], 2.0)
        assert np.allclose(result.vectors[2], 3.0)

        # Check get_vector_by_id lookup
        assert np.allclose(result.get_vector_by_id("doc.pdf#chunk_1"), 2.0)
        assert result.get_vector_by_id("missing_id") is None

        # Check metadata preservation separately
        assert result.metadata[1] == {
            "chunk_id": "doc.pdf#chunk_1",
            "source_filename": "doc.pdf",
            "chunk_index": 1,
            "page_numbers": [1, 2],
        }

        # Check to_embedded_chunks helper
        embedded_chunks = result.to_embedded_chunks()
        assert len(embedded_chunks) == 3
        assert isinstance(embedded_chunks[0], EmbeddedChunk)
        assert embedded_chunks[0].chunk_id == "doc.pdf#chunk_0"
        assert embedded_chunks[0].text == "Text zero"
        assert np.allclose(embedded_chunks[0].vector, 1.0)

    def test_batching_mechanism(self) -> None:
        """Verifies that texts are chunked into batches and encoded sequentially."""
        chunks = _create_sample_chunks(5)
        batch_size = 2

        mock_model = MagicMock()

        def mock_encode(batch_slice, **kwargs):
            return np.ones((len(batch_slice), 384), dtype=np.float32)

        mock_model.encode.side_effect = mock_encode

        embedder = SentenceTransformerEmbedder(
            model=mock_model, batch_size=batch_size, dimension=384
        )
        result = embedder.embed_chunks(chunks)

        # 5 items with batch_size 2 => 3 batches (2, 2, 1)
        assert mock_model.encode.call_count == 3
        first_call_args = mock_model.encode.call_args_list[0][0][0]
        second_call_args = mock_model.encode.call_args_list[1][0][0]
        third_call_args = mock_model.encode.call_args_list[2][0][0]

        assert len(first_call_args) == 2
        assert len(second_call_args) == 2
        assert len(third_call_args) == 1

        assert result.vectors.shape == (5, 384)

    def test_invalid_numerical_output_nan(self) -> None:
        """Verifies that non-finite values (NaN) raise ValueError."""
        chunks = _create_sample_chunks(2)
        mock_model = MagicMock()
        corrupt_vectors = np.ones((2, 384), dtype=np.float32)
        corrupt_vectors[0, 5] = np.nan
        mock_model.encode.return_value = corrupt_vectors

        embedder = SentenceTransformerEmbedder(model=mock_model, dimension=384)
        with pytest.raises(ValueError, match="non-finite values"):
            embedder.embed_chunks(chunks)

    def test_invalid_numerical_output_inf(self) -> None:
        """Verifies that infinite values (Inf) raise ValueError."""
        chunks = _create_sample_chunks(2)
        mock_model = MagicMock()
        corrupt_vectors = np.ones((2, 384), dtype=np.float32)
        corrupt_vectors[1, 10] = np.inf
        mock_model.encode.return_value = corrupt_vectors

        embedder = SentenceTransformerEmbedder(model=mock_model, dimension=384)
        with pytest.raises(ValueError, match="non-finite values"):
            embedder.embed_chunks(chunks)

    def test_dimension_mismatch_raises_error(self) -> None:
        """Verifies that vector dimension mismatch raises ValueError."""
        mock_model = MagicMock()
        wrong_dim_vectors = np.ones((2, 128), dtype=np.float32)
        mock_model.encode.return_value = wrong_dim_vectors

        embedder = SentenceTransformerEmbedder(model=mock_model, dimension=384)
        with pytest.raises(ValueError, match="dimension mismatch"):
            embedder.embed_texts(["Text 1", "Text 2"])

    def test_vector_count_mismatch_raises_error(self) -> None:
        """Verifies that vector count mismatch raises ValueError."""
        mock_model = MagicMock()
        fewer_vectors = np.ones((1, 384), dtype=np.float32)
        mock_model.encode.return_value = fewer_vectors

        embedder = SentenceTransformerEmbedder(model=mock_model, dimension=384)
        with pytest.raises(ValueError, match="count mismatch"):
            embedder.embed_texts(["Text 1", "Text 2"])

    def test_invalid_batch_size_raises_error(self) -> None:
        """Verifies that non-positive batch_size raises ValueError."""
        mock_model = MagicMock()
        with pytest.raises(ValueError, match="batch_size must be positive"):
            SentenceTransformerEmbedder(model=mock_model, batch_size=0)

        embedder = SentenceTransformerEmbedder(model=mock_model, batch_size=16)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            embedder.embed_texts(["Hello"], batch_size=-5)

    def test_model_reuse_avoid_repeated_loading(self) -> None:
        """Verifies get_embedder caches and reuses the model instance without reloading."""
        reset_embedder()
        mock_st_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_instance.encode.return_value = np.ones((1, 384), dtype=np.float32)
        mock_st_class.return_value = mock_instance

        with patch(
            "sentence_transformers.SentenceTransformer", mock_st_class
        ):
            # First call creates instance
            embedder1 = get_embedder()
            # Second call retrieves cached instance
            embedder2 = get_embedder()

            assert embedder1 is embedder2
            assert mock_st_class.call_count == 1

            # Convenience embed_chunks also reuses the shared embedder
            chunks = _create_sample_chunks(1)
            embed_chunks(chunks)
            assert mock_st_class.call_count == 1

        reset_embedder()
