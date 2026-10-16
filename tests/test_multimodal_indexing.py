"""Comprehensive tests for Day 13 modality-specific multimodal FAISS indexing."""

import io
from typing import Any, List, Optional
import numpy as np
import pytest
from PIL import Image

from app.chunking.models import TextChunk
from app.embeddings.base import BaseEmbedder
from app.indexing import (
    ModalityFAISSIndex,
    MultimodalFAISSIndex,
    MultimodalSearchResult,
    build_multimodal_index,
)
from app.multimodal import (
    CaptionMatch,
    ElementType,
    ExtractedCaption,
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
    MultimodalDocument,
    MultimodalElement,
    MultimodalEmbedder,
    assemble_multimodal_document,
)
from app.multimodal.embedding import (
    BaseVisualEmbedder,
    MultimodalEmbeddedElement,
    MultimodalEmbeddingResult,
)


# ===========================================================================
# Test Helpers and Mocks
# ===========================================================================


def _create_sample_png_bytes(color: str = "blue", size: tuple[int, int] = (16, 16)) -> bytes:
    """Creates in-memory PNG image bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class MockTextEmbedder(BaseEmbedder):
    """Deterministic 384-dimensional mock text embedder."""

    @property
    def dimension(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return "mock-minilm-384"

    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        vectors = []
        for idx, t in enumerate(texts):
            vec = np.zeros(384, dtype=np.float32)
            # deterministic signature: unit length along a distinct coordinate
            coord = idx % 384
            vec[coord] = 1.0
            vectors.append(vec)
        return np.vstack(vectors).astype(np.float32)


class MockVisualEmbedder(BaseVisualEmbedder):
    """Deterministic 512-dimensional mock visual embedder."""

    @property
    def dimension(self) -> int:
        return 512

    @property
    def model_name(self) -> str:
        return "mock-clip-512"

    def embed_images(self, images: List[Any], batch_size: Optional[int] = None) -> np.ndarray:
        if not images:
            return np.empty((0, 512), dtype=np.float32)
        vectors = []
        for idx, _ in enumerate(images):
            vec = np.zeros(512, dtype=np.float32)
            coord = idx % 512
            vec[coord] = 1.0
            vectors.append(vec)
        return np.vstack(vectors).astype(np.float32)


def _create_sample_multimodal_element(
    element_id: str,
    element_type: ElementType,
    page_number: int = 1,
    content: Any = "Sample content",
    dimension: Optional[int] = None,
    related_element_ids: Optional[List[str]] = None,
    metadata: Optional[dict] = None,
) -> MultimodalElement:
    """Helper creating a sample MultimodalElement with an optional synthetic embedding."""
    embedding = None
    if dimension is not None:
        embedding = np.zeros(dimension, dtype=np.float32)
        embedding[0] = 1.0

    return MultimodalElement(
        element_id=element_id,
        element_type=element_type,
        source_filename="test_paper.pdf",
        page_number=page_number,
        content=content,
        embedding=embedding,
        related_element_ids=related_element_ids or [],
        metadata=metadata or {},
    )


# ===========================================================================
# Category A: Dimensions
# ===========================================================================


class TestDimensions:
    """Verifies that text index is 384-d and visual index is 512-d."""

    def test_default_dimensions(self) -> None:
        index = MultimodalFAISSIndex()
        assert index.text_dimension == 384
        assert index.text_index.dimension == 384
        assert index.visual_dimension == 512
        assert index.visual_index.dimension == 512

    def test_custom_dimensions(self) -> None:
        index = MultimodalFAISSIndex(text_dimension=128, visual_dimension=256)
        assert index.text_dimension == 128
        assert index.text_index.dimension == 128
        assert index.visual_dimension == 256
        assert index.visual_index.dimension == 256

    def test_invalid_dimension_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension must be a positive integer"):
            ModalityFAISSIndex(dimension=0)

        with pytest.raises(ValueError, match="dimension must be a positive integer"):
            ModalityFAISSIndex(dimension=-5)


# ===========================================================================
# Category B: Dimension Safety
# ===========================================================================


class TestDimensionSafety:
    """Verifies strict dimensional isolation and rejection of cross-modal vectors."""

    def test_512d_vector_rejected_by_text_index(self) -> None:
        index = MultimodalFAISSIndex()
        elem = _create_sample_multimodal_element("text_1", ElementType.TEXT)
        vec_512 = np.ones((1, 512), dtype=np.float32)

        with pytest.raises(ValueError, match="Vector dimension 512 does not match index dimension 384"):
            index.text_index.add_elements([elem], vectors=vec_512)

    def test_384d_vector_rejected_by_visual_index(self) -> None:
        index = MultimodalFAISSIndex()
        elem = _create_sample_multimodal_element("img_1", ElementType.IMAGE, content=b"fake")
        vec_384 = np.ones((1, 384), dtype=np.float32)

        with pytest.raises(ValueError, match="Vector dimension 384 does not match index dimension 512"):
            index.visual_index.add_elements([elem], vectors=vec_384)

    def test_query_dimension_mismatch_rejected(self) -> None:
        index = MultimodalFAISSIndex()
        elem = _create_sample_multimodal_element("text_1", ElementType.TEXT, dimension=384)
        index.text_index.add_elements([elem])

        wrong_query = np.ones((512,), dtype=np.float32)
        with pytest.raises(ValueError, match="query_vector dimension \\(512\\) does not match index dimension \\(384\\)"):
            index.text_index.search_by_vector(wrong_query)


# ===========================================================================
# Category C: Modality Routing
# ===========================================================================


class TestModalityRouting:
    """Verifies text/caption/table/equation route to text index, and image routes to visual index."""

    def test_routing_via_embedding_result(self) -> None:
        index = MultimodalFAISSIndex()

        # Create elements of all 5 types
        text_elem = _create_sample_multimodal_element("txt_1", ElementType.TEXT)
        cap_elem = _create_sample_multimodal_element("cap_1", ElementType.CAPTION)
        tbl_elem = _create_sample_multimodal_element("tbl_1", ElementType.TABLE)
        eq_elem = _create_sample_multimodal_element("eq_1", ElementType.EQUATION)
        img_elem = _create_sample_multimodal_element("img_1", ElementType.IMAGE, content=b"bytes")

        # 384-d vectors for text-like, 512-d for visual
        vec_384 = np.ones(384, dtype=np.float32)
        vec_512 = np.ones(512, dtype=np.float32)

        embedded_elements = [
            MultimodalEmbeddedElement("txt_1", ElementType.TEXT, "paper.pdf", 1, vec_384, text_elem),
            MultimodalEmbeddedElement("cap_1", ElementType.CAPTION, "paper.pdf", 1, vec_384, cap_elem),
            MultimodalEmbeddedElement("tbl_1", ElementType.TABLE, "paper.pdf", 2, vec_384, tbl_elem),
            MultimodalEmbeddedElement("eq_1", ElementType.EQUATION, "paper.pdf", 2, vec_384, eq_elem),
            MultimodalEmbeddedElement("img_1", ElementType.IMAGE, "paper.pdf", 3, vec_512, img_elem),
        ]
        emb_result = MultimodalEmbeddingResult(
            elements=[text_elem, cap_elem, tbl_elem, eq_elem, img_elem],
            embedded_elements=embedded_elements,
        )

        index.add_embedding_result(emb_result)

        # 4 text-like elements in text_index
        assert index.total_text_elements == 4
        # 1 image in visual_index
        assert index.total_visual_elements == 1
        assert index.total_elements == 5
        assert len(index) == 5

        # Verify element IDs are in their respective indices
        assert index.text_index.get_element_by_element_id("txt_1") is not None
        assert index.text_index.get_element_by_element_id("cap_1") is not None
        assert index.text_index.get_element_by_element_id("tbl_1") is not None
        assert index.text_index.get_element_by_element_id("eq_1") is not None
        assert index.text_index.get_element_by_element_id("img_1") is None

        assert index.visual_index.get_element_by_element_id("img_1") is not None
        assert index.visual_index.get_element_by_element_id("txt_1") is None


# ===========================================================================
# Category D: FAISS ID Mapping
# ===========================================================================


class TestFAISSIDMapping:
    """Verifies bidirectional mapping between FAISS internal integer IDs and MultimodalElement."""

    def test_internal_integer_id_mapping(self) -> None:
        idx = ModalityFAISSIndex(dimension=384)
        elems = [
            _create_sample_multimodal_element(f"elem_{i}", ElementType.TEXT, dimension=384)
            for i in range(3)
        ]
        idx.add_elements(elems)

        assert idx.total_elements == 3

        # Recover elements by 0-based integer ID
        for i in range(3):
            recovered = idx.get_element_by_id(i)
            assert recovered is not None
            assert recovered.element_id == f"elem_{i}"
            assert idx.get_internal_id_by_element_id(f"elem_{i}") == i

        # Missing IDs return None
        assert idx.get_element_by_id(99) is None
        assert idx.get_element_by_element_id("non_existent") is None
        assert idx.get_internal_id_by_element_id("non_existent") is None


# ===========================================================================
# Category E: Retrieval & Ranking
# ===========================================================================


class TestRetrievalAndRanking:
    """Verifies cosine similarity calculation, ranking, top_k, and empty indices."""

    def test_empty_index_returns_empty_list(self) -> None:
        idx = MultimodalFAISSIndex()
        query_384 = np.ones((384,), dtype=np.float32)
        query_512 = np.ones((512,), dtype=np.float32)

        assert idx.search_text_by_vector(query_384) == []
        assert idx.search_visual_by_vector(query_512) == []
        assert idx.search_text("hello world") == []
        assert idx.search_text("") == []
        assert idx.search_text("   ") == []

    def test_cosine_similarity_ranking_and_scores(self) -> None:
        idx = ModalityFAISSIndex(dimension=4)

        # 3 elements in 4D:
        # e0: identical direction to query [1, 0, 0, 0] -> score = 1.0
        # e1: 45° angle to query [1, 1, 0, 0] -> score = cos(45°) ~ 0.7071
        # e2: orthogonal to query [0, 1, 0, 0] -> score = 0.0
        e0 = _create_sample_multimodal_element("e0", ElementType.TEXT)
        e1 = _create_sample_multimodal_element("e1", ElementType.TEXT)
        e2 = _create_sample_multimodal_element("e2", ElementType.TEXT)

        vectors = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        idx.add_elements([e0, e1, e2], vectors=vectors)

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = idx.search_by_vector(query, top_k=3)

        assert len(results) == 3
        # Check ordering: e0 (1.0) > e1 (~0.7071) > e2 (0.0)
        assert results[0].element_id == "e0"
        assert pytest.approx(results[0].score, abs=1e-5) == 1.0

        assert results[1].element_id == "e1"
        assert pytest.approx(results[1].score, abs=1e-4) == 0.7071

        assert results[2].element_id == "e2"
        assert pytest.approx(results[2].score, abs=1e-5) == 0.0

    def test_top_k_limiting(self) -> None:
        idx = ModalityFAISSIndex(dimension=4)
        elems = [
            _create_sample_multimodal_element(f"e_{i}", ElementType.TEXT)
            for i in range(10)
        ]
        vectors = np.eye(10, 4, dtype=np.float32)
        idx.add_elements(elems, vectors=vectors)

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        top_2 = idx.search_by_vector(query, top_k=2)
        assert len(top_2) == 2

        top_20 = idx.search_by_vector(query, top_k=20)
        assert len(top_20) == 10  # capped at total_elements


# ===========================================================================
# Category F: Provenance & Relationship Preservation
# ===========================================================================


class TestProvenancePreservation:
    """Verifies that all provenance fields and relationship IDs are preserved in search results."""

    def test_multimodal_search_result_fields(self) -> None:
        idx = ModalityFAISSIndex(dimension=4)
        metadata = {"bbox": [10.0, 20.0, 100.0, 200.0], "format": "figure"}
        related_ids = ["caption_42", "text_99"]

        elem = MultimodalElement(
            element_id="img_100",
            element_type=ElementType.IMAGE,
            source_filename="transformer.pdf",
            page_number=5,
            content=b"fake_image_bytes",
            related_element_ids=related_ids,
            metadata=metadata,
        )
        vec = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        idx.add_elements([elem], vectors=vec)

        query = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        results = idx.search_by_vector(query, top_k=1)

        assert len(results) == 1
        hit = results[0]

        assert isinstance(hit, MultimodalSearchResult)
        assert hit.element is elem
        assert hit.element_id == "img_100"
        assert hit.element_type == ElementType.IMAGE
        assert hit.source_filename == "transformer.pdf"
        assert hit.page_number == 5
        assert hit.related_element_ids == ["caption_42", "text_99"]
        assert hit.metadata == metadata
        assert hit.content == b"fake_image_bytes"
        assert pytest.approx(hit.score, abs=1e-5) == 1.0


# ===========================================================================
# Category G: Validation and Edge Cases
# ===========================================================================


class TestValidationAndEdgeCases:
    """Verifies rejection of invalid inputs (NaN, Inf, top_k <= 0, shape mismatches)."""

    def test_nan_vector_rejected(self) -> None:
        idx = ModalityFAISSIndex(dimension=4)
        elem = _create_sample_multimodal_element("e_nan", ElementType.TEXT)
        nan_vec = np.array([[1.0, np.nan, 0.0, 0.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="vectors contain non-finite values"):
            idx.add_elements([elem], vectors=nan_vec)

    def test_inf_query_vector_rejected(self) -> None:
        idx = ModalityFAISSIndex(dimension=4)
        elem = _create_sample_multimodal_element("e1", ElementType.TEXT)
        idx.add_elements([elem], vectors=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))

        inf_query = np.array([1.0, np.inf, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="query_vector contains non-finite values"):
            idx.search_by_vector(inf_query)

    def test_invalid_top_k_rejected(self) -> None:
        idx = ModalityFAISSIndex(dimension=4)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            idx.search_by_vector(query, top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            idx.search_by_vector(query, top_k=-2)

    def test_invalid_query_vector_shape_rejected(self) -> None:
        idx = ModalityFAISSIndex(dimension=4)
        elem = _create_sample_multimodal_element("e1", ElementType.TEXT)
        idx.add_elements([elem], vectors=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))

        # 3D array
        with pytest.raises(ValueError, match="query_vector must have shape"):
            idx.search_by_vector(np.ones((1, 1, 4), dtype=np.float32))

        # 2D array with multiple rows
        with pytest.raises(ValueError, match="query_vector must have shape"):
            idx.search_by_vector(np.ones((2, 4), dtype=np.float32))

    def test_empty_elements_add_is_noop(self) -> None:
        idx = ModalityFAISSIndex(dimension=4)
        idx.add_elements([])
        assert idx.total_elements == 0


# ===========================================================================
# Category H: Mixed Multimodal Document Integration
# ===========================================================================


class TestMixedMultimodalDocument:
    """Verifies end-to-end integration: assembling a mixed document, embedding, and indexing."""

    def test_assemble_embed_and_index_mixed_document(self) -> None:
        png_data = _create_sample_png_bytes(color="green")

        chunk = TextChunk(
            chunk_id="chunk_1",
            text="Attention is all you need for sequence transduction.",
            source_filename="paper.pdf",
            chunk_index=0,
            page_numbers=[1],
        )
        img = ExtractedImage(
            image_id="img_1",
            source_filename="paper.pdf",
            page_number=2,
            image_index=0,
            image_format="png",
            image_data=png_data,
        )
        cap = ExtractedCaption(
            caption_id="cap_1",
            source_filename="paper.pdf",
            page_number=2,
            caption_index=0,
            caption_type="figure",
            label="Figure 1",
            text="The Transformer model architecture.",
        )
        tbl = ExtractedTable(
            table_id="tbl_1",
            source_filename="paper.pdf",
            page_number=3,
            table_index=0,
            num_rows=2,
            num_cols=2,
            content="| Model | BLEU |\n|---|---|\n| Base | 27.3 |",
        )
        eq = ExtractedEquation(
            equation_id="eq_1",
            source_filename="paper.pdf",
            page_number=3,
            equation_index=0,
            text=r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
        )
        match = CaptionMatch(
            element_id="img_1",
            caption_id="cap_1",
            caption_type="figure",
            proximity_pts=15.0,
            source_filename="paper.pdf",
            page_number=2,
        )

        doc = assemble_multimodal_document(
            text_chunks=[chunk],
            images=[img],
            captions=[cap],
            tables=[tbl],
            equations=[eq],
            caption_matches=[match],
            source_filename="paper.pdf",
        )

        assert len(doc) == 5

        # Embed using deterministic mock embedders
        embedder = MultimodalEmbedder(
            text_embedder=MockTextEmbedder(),
            visual_embedder=MockVisualEmbedder(),
        )

        # Build multimodal index directly from document
        index = build_multimodal_index(doc, embedder=embedder)

        # Verify correct routing
        assert index.total_text_elements == 4  # chunk, cap, tbl, eq
        assert index.total_visual_elements == 1  # img
        assert index.total_elements == 5

        # Test text search with query string
        text_results = index.search_text(
            query="Transformer architecture",
            top_k=3,
            embedder=embedder,
        )
        assert len(text_results) == 3
        # All retrieved items must be text-like
        for r in text_results:
            assert r.element_type in {
                ElementType.TEXT,
                ElementType.CAPTION,
                ElementType.TABLE,
                ElementType.EQUATION,
            }

        # Test visual search with image query
        visual_results = index.search_visual(
            query=png_data,
            top_k=2,
            embedder=embedder,
        )
        assert len(visual_results) == 1
        assert visual_results[0].element_id == "img_1"
        assert visual_results[0].element_type == ElementType.IMAGE

        # Test relationship traversal from search result
        img_hit = visual_results[0]
        assert "cap_1" in img_hit.related_element_ids

        # Look up related caption from the document
        related = doc.get_related(img_hit.element_id)
        assert len(related) == 1
        assert related[0].element_id == "cap_1"
        assert related[0].element_type == ElementType.CAPTION
        assert "The Transformer model architecture" in related[0].text
