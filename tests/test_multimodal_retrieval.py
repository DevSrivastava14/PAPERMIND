"""Focused tests for Day 13 Step 3: modality-aware multimodal retrieval."""

from typing import Any, List, Optional
import numpy as np
import pytest

from app.chunking.models import TextChunk
from app.embeddings.base import BaseEmbedder
from app.indexing import (
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
from app.retrieval import (
    MultimodalRetrievalPipeline,
    build_multimodal_pipeline,
)


# ===========================================================================
# Deterministic Mock Embedders
# ===========================================================================


class MockTextEmbedder(BaseEmbedder):
    """Deterministic 384-d mock text embedder."""

    @property
    def dimension(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return "mock-text-384"

    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        vectors = []
        for idx, t in enumerate(texts):
            vec = np.zeros(384, dtype=np.float32)
            # Use hash or length for deterministic orientation
            dim_idx = (len(t) * 7 + idx) % 384
            vec[dim_idx] = 1.0
            vectors.append(vec)
        return np.vstack(vectors).astype(np.float32)


class MockVisualEmbedder(BaseVisualEmbedder):
    """Deterministic 512-d mock visual embedder."""

    @property
    def dimension(self) -> int:
        return 512

    @property
    def model_name(self) -> str:
        return "mock-visual-512"

    def embed_images(self, images: List[Any], batch_size: Optional[int] = None) -> np.ndarray:
        if not images:
            return np.empty((0, 512), dtype=np.float32)
        vectors = []
        for idx, _ in enumerate(images):
            vec = np.zeros(512, dtype=np.float32)
            vec[idx % 512] = 1.0
            vectors.append(vec)
        return np.vstack(vectors).astype(np.float32)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def mock_text_embedder() -> MockTextEmbedder:
    return MockTextEmbedder()


@pytest.fixture
def mock_visual_embedder() -> MockVisualEmbedder:
    return MockVisualEmbedder()


@pytest.fixture
def sample_multimodal_index() -> MultimodalFAISSIndex:
    """Creates a pre-populated MultimodalFAISSIndex with synthetic text and visual elements."""
    index = MultimodalFAISSIndex()

    # Create 3 text-like elements
    text_el = MultimodalElement(
        element_id="txt_01",
        element_type=ElementType.TEXT,
        source_filename="research_paper.pdf",
        page_number=1,
        content="Transformer architecture uses multi-head attention.",
        metadata={"section": "introduction"},
    )
    cap_el = MultimodalElement(
        element_id="cap_01",
        element_type=ElementType.CAPTION,
        source_filename="research_paper.pdf",
        page_number=2,
        content="Figure 1: Full encoder-decoder schematic.",
        related_element_ids=["img_01"],
        metadata={"label": "Figure 1"},
    )
    tbl_el = MultimodalElement(
        element_id="tbl_01",
        element_type=ElementType.TABLE,
        source_filename="research_paper.pdf",
        page_number=3,
        content="| BLEU | 28.4 |",
        metadata={"rows": 1, "cols": 2},
    )

    # 384-d vectors with deliberate angles:
    # txt_01: unit vector along coordinate 0
    # cap_01: vector in plane (0, 1) with 45 degrees
    # tbl_01: unit vector along coordinate 1
    v_txt = np.zeros(384, dtype=np.float32)
    v_txt[0] = 1.0

    v_cap = np.zeros(384, dtype=np.float32)
    v_cap[0] = 1.0
    v_cap[1] = 1.0

    v_tbl = np.zeros(384, dtype=np.float32)
    v_tbl[1] = 1.0

    # Create 2 visual elements
    img1_el = MultimodalElement(
        element_id="img_01",
        element_type=ElementType.IMAGE,
        source_filename="research_paper.pdf",
        page_number=2,
        content=b"\x89PNGfakeimage1",
        related_element_ids=["cap_01"],
        metadata={"width": 400, "height": 300},
    )
    img2_el = MultimodalElement(
        element_id="img_02",
        element_type=ElementType.IMAGE,
        source_filename="research_paper.pdf",
        page_number=4,
        content=b"\x89PNGfakeimage2",
        metadata={"width": 600, "height": 450},
    )

    v_img1 = np.zeros(512, dtype=np.float32)
    v_img1[0] = 1.0

    v_img2 = np.zeros(512, dtype=np.float32)
    v_img2[5] = 1.0

    emb_elements = [
        MultimodalEmbeddedElement("txt_01", ElementType.TEXT, "research_paper.pdf", 1, v_txt, text_el),
        MultimodalEmbeddedElement("cap_01", ElementType.CAPTION, "research_paper.pdf", 2, v_cap, cap_el),
        MultimodalEmbeddedElement("tbl_01", ElementType.TABLE, "research_paper.pdf", 3, v_tbl, tbl_el),
        MultimodalEmbeddedElement("img_01", ElementType.IMAGE, "research_paper.pdf", 2, v_img1, img1_el),
        MultimodalEmbeddedElement("img_02", ElementType.IMAGE, "research_paper.pdf", 4, v_img2, img2_el),
    ]

    result = MultimodalEmbeddingResult(
        elements=[text_el, cap_el, tbl_el, img1_el, img2_el],
        embedded_elements=emb_elements,
    )
    index.add_embedding_result(result)
    return index


# ===========================================================================
# Category A: Text Multimodal Retrieval
# ===========================================================================


class TestTextMultimodalRetrieval:
    """Verifies text-like multimodal retrieval, ranking, and element return."""

    def test_retrieve_text_by_vector_ranking(
        self,
        sample_multimodal_index: MultimodalFAISSIndex,
        mock_text_embedder: MockTextEmbedder,
        mock_visual_embedder: MockVisualEmbedder,
    ) -> None:
        pipeline = build_multimodal_pipeline(
            index=sample_multimodal_index,
            text_embedder=mock_text_embedder,
            visual_embedder=mock_visual_embedder,
            default_top_k=3,
        )

        # Query vector along coordinate 0:
        # matches txt_01 (score = 1.0), cap_01 (score ~ 0.7071), tbl_01 (score = 0.0)
        query = np.zeros(384, dtype=np.float32)
        query[0] = 1.0

        results = pipeline.retrieve_text(query=query, top_k=3)

        assert len(results) == 3
        # Check ranking
        assert results[0].element_id == "txt_01"
        assert pytest.approx(results[0].score, abs=1e-4) == 1.0
        assert results[0].element.content == "Transformer architecture uses multi-head attention."

        assert results[1].element_id == "cap_01"
        assert pytest.approx(results[1].score, abs=1e-4) == 0.7071
        assert results[1].element_type == ElementType.CAPTION

        assert results[2].element_id == "tbl_01"
        assert pytest.approx(results[2].score, abs=1e-4) == 0.0

    def test_retrieve_text_by_string_query(
        self,
        sample_multimodal_index: MultimodalFAISSIndex,
        mock_text_embedder: MockTextEmbedder,
    ) -> None:
        pipeline = MultimodalRetrievalPipeline(
            index=sample_multimodal_index,
            text_embedder=mock_text_embedder,
        )

        results = pipeline.retrieve_text("Sample attention mechanism", top_k=2)
        assert len(results) <= 2
        for r in results:
            assert isinstance(r, MultimodalSearchResult)
            assert isinstance(r.element, MultimodalElement)


# ===========================================================================
# Category B: Visual Retrieval
# ===========================================================================


class TestVisualRetrieval:
    """Verifies visual retrieval from the visual index."""

    def test_retrieve_visual_by_vector(
        self,
        sample_multimodal_index: MultimodalFAISSIndex,
        mock_text_embedder: MockTextEmbedder,
        mock_visual_embedder: MockVisualEmbedder,
    ) -> None:
        pipeline = MultimodalRetrievalPipeline(
            index=sample_multimodal_index,
            text_embedder=mock_text_embedder,
            visual_embedder=mock_visual_embedder,
        )

        # Query vector along coordinate 0: matches img_01 perfectly
        query = np.zeros(512, dtype=np.float32)
        query[0] = 1.0

        results = pipeline.retrieve_visual(query=query, top_k=1)
        assert len(results) == 1
        assert results[0].element_id == "img_01"
        assert results[0].element_type == ElementType.IMAGE
        assert pytest.approx(results[0].score, abs=1e-4) == 1.0
        assert results[0].content == b"\x89PNGfakeimage1"

    def test_retrieve_visual_by_bytes(
        self,
        sample_multimodal_index: MultimodalFAISSIndex,
        mock_visual_embedder: MockVisualEmbedder,
    ) -> None:
        pipeline = MultimodalRetrievalPipeline(
            index=sample_multimodal_index,
            visual_embedder=mock_visual_embedder,
        )

        results = pipeline.retrieve_visual(b"raw_image_query_bytes", top_k=2)
        assert len(results) == 2
        for r in results:
            assert r.element_type == ElementType.IMAGE


# ===========================================================================
# Category C: Modality Isolation
# ===========================================================================


class TestModalityIsolation:
    """Verifies that text queries never return images and visual queries never return text."""

    def test_text_retrieval_never_returns_images(
        self,
        sample_multimodal_index: MultimodalFAISSIndex,
        mock_text_embedder: MockTextEmbedder,
    ) -> None:
        pipeline = MultimodalRetrievalPipeline(
            index=sample_multimodal_index,
            text_embedder=mock_text_embedder,
        )

        # Retrieve all available results in text stream
        results = pipeline.retrieve_text(np.ones(384, dtype=np.float32), top_k=10)
        assert len(results) == 3
        for r in results:
            assert r.element_type != ElementType.IMAGE
            assert r.element_type in {
                ElementType.TEXT,
                ElementType.CAPTION,
                ElementType.TABLE,
                ElementType.EQUATION,
            }

    def test_visual_retrieval_never_returns_text_like(
        self,
        sample_multimodal_index: MultimodalFAISSIndex,
        mock_visual_embedder: MockVisualEmbedder,
    ) -> None:
        pipeline = MultimodalRetrievalPipeline(
            index=sample_multimodal_index,
            visual_embedder=mock_visual_embedder,
        )

        results = pipeline.retrieve_visual(np.ones(512, dtype=np.float32), top_k=10)
        assert len(results) == 2
        for r in results:
            assert r.element_type == ElementType.IMAGE


# ===========================================================================
# Category D: Provenance Preservation
# ===========================================================================


class TestProvenancePreservation:
    """Verifies exact preservation of ID, source, 1-based page number, type, and relationships."""

    def test_provenance_integrity(
        self,
        sample_multimodal_index: MultimodalFAISSIndex,
    ) -> None:
        pipeline = MultimodalRetrievalPipeline(index=sample_multimodal_index)

        query = np.zeros(384, dtype=np.float32)
        query[0] = 1.0

        results = pipeline.retrieve_text(query=query, top_k=2)
        hit = results[0]

        assert hit.element_id == "txt_01"
        assert hit.source_filename == "research_paper.pdf"
        assert hit.page_number == 1
        assert isinstance(hit.page_number, int)
        assert hit.page_number >= 1
        assert hit.element_type == ElementType.TEXT
        assert hit.metadata == {"section": "introduction"}


# ===========================================================================
# Category E: Relationship Preservation
# ===========================================================================


class TestRelationshipPreservation:
    """Verifies resolving inter-element relations (image <-> caption) from retrieval hits."""

    def test_image_retrieval_allows_caption_resolution(
        self,
        mock_text_embedder: MockTextEmbedder,
        mock_visual_embedder: MockVisualEmbedder,
    ) -> None:
        chunk = TextChunk(
            chunk_id="chunk_1",
            text="Text mentioning Figure 1 architecture.",
            source_filename="paper.pdf",
            chunk_index=0,
            page_numbers=[1],
        )
        img = ExtractedImage(
            image_id="img_arch",
            source_filename="paper.pdf",
            page_number=2,
            image_index=0,
            image_format="png",
            image_data=b"\x89PNGdemo",
        )
        cap = ExtractedCaption(
            caption_id="cap_arch",
            source_filename="paper.pdf",
            page_number=2,
            caption_index=0,
            caption_type="figure",
            label="Figure 1",
            text="Figure 1: Overall system flowchart.",
        )
        match = CaptionMatch(
            element_id="img_arch",
            caption_id="cap_arch",
            caption_type="figure",
            proximity_pts=10.0,
            source_filename="paper.pdf",
            page_number=2,
        )

        doc = assemble_multimodal_document(
            text_chunks=[chunk],
            images=[img],
            captions=[cap],
            caption_matches=[match],
            source_filename="paper.pdf",
        )

        embedder = MultimodalEmbedder(
            text_embedder=mock_text_embedder,
            visual_embedder=mock_visual_embedder,
        )
        index = build_multimodal_index(doc, embedder=embedder)
        pipeline = build_multimodal_pipeline(
            index=index,
            text_embedder=mock_text_embedder,
            visual_embedder=mock_visual_embedder,
        )

        # Retrieve visual element
        img_results = pipeline.retrieve_visual(b"\x89PNGdemo", top_k=1)
        assert len(img_results) == 1
        img_hit = img_results[0]

        # Verify related_element_ids is populated
        assert img_hit.related_element_ids == ["cap_arch"]

        # Resolve related caption through document
        related = doc.get_related(img_hit.element_id)
        assert len(related) == 1
        assert related[0].element_id == "cap_arch"
        assert related[0].element_type == ElementType.CAPTION
        assert "Figure 1: Overall system flowchart." in related[0].text


# ===========================================================================
# Category F: Validation & Error Handling
# ===========================================================================


class TestValidationAndErrorHandling:
    """Verifies handling of invalid queries, dimensions, top_k, and modality selection."""

    def test_empty_string_query_returns_empty_list(
        self,
        sample_multimodal_index: MultimodalFAISSIndex,
        mock_text_embedder: MockTextEmbedder,
    ) -> None:
        pipeline = MultimodalRetrievalPipeline(
            index=sample_multimodal_index,
            text_embedder=mock_text_embedder,
        )
        assert pipeline.retrieve_text("") == []
        assert pipeline.retrieve_text("   ") == []

    def test_invalid_top_k_raises_error(
        self,
        sample_multimodal_index: MultimodalFAISSIndex,
    ) -> None:
        pipeline = MultimodalRetrievalPipeline(index=sample_multimodal_index)
        query = np.ones(384, dtype=np.float32)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            pipeline.retrieve_text(query, top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            pipeline.retrieve_visual(np.ones(512, dtype=np.float32), top_k=-1)

    def test_dimension_mismatch_raises_error(
        self,
        sample_multimodal_index: MultimodalFAISSIndex,
    ) -> None:
        pipeline = MultimodalRetrievalPipeline(index=sample_multimodal_index)

        # 512-d vector passed to text retrieval (expects 384-d)
        with pytest.raises(ValueError, match="query_vector dimension \\(512\\) does not match index dimension \\(384\\)"):
            pipeline.retrieve_text(np.ones(512, dtype=np.float32))

        # 384-d vector passed to visual retrieval (expects 512-d)
        with pytest.raises(ValueError, match="query_vector dimension \\(384\\) does not match index dimension \\(512\\)"):
            pipeline.retrieve_visual(np.ones(384, dtype=np.float32))

    def test_generic_retrieve_modality_dispatch(
        self,
        sample_multimodal_index: MultimodalFAISSIndex,
    ) -> None:
        pipeline = MultimodalRetrievalPipeline(index=sample_multimodal_index)

        text_vec = np.zeros(384, dtype=np.float32)
        text_vec[0] = 1.0

        visual_vec = np.zeros(512, dtype=np.float32)
        visual_vec[0] = 1.0

        # Dispatch 'text'
        t_res = pipeline.retrieve(text_vec, modality="text", top_k=2)
        assert len(t_res) == 2
        assert t_res[0].element_type != ElementType.IMAGE

        # Dispatch 'visual'
        v_res = pipeline.retrieve(visual_vec, modality="visual", top_k=1)
        assert len(v_res) == 1
        assert v_res[0].element_type == ElementType.IMAGE

        # Dispatch invalid modality raises ValueError
        with pytest.raises(ValueError, match="Unsupported modality 'audio'"):
            pipeline.retrieve(text_vec, modality="audio")

    def test_pipeline_init_validation(self) -> None:
        with pytest.raises(TypeError, match="index must be an instance of MultimodalFAISSIndex"):
            MultimodalRetrievalPipeline(index="not_an_index")  # type: ignore

        with pytest.raises(ValueError, match="default_top_k must be a positive integer"):
            MultimodalRetrievalPipeline(index=MultimodalFAISSIndex(), default_top_k=0)
