"""Focused tests for Day 15 Task 2: BM25 sparse retriever integration into MultimodalRetrievalPipeline.

Verifies:
- Successful sparse retrieval via MultimodalRetrievalPipeline.retrieve_sparse
- Correct BM25 delegation (retrieve_sparse, search_sparse alias, retrieve(..., modality='sparse'))
- Text-like element retrieval: TEXT, CAPTION, TABLE, EQUATION
- Strict IMAGE exclusion during sparse retrieval
- top_k propagation (custom top_k, default_top_k fallback)
- Complete provenance preservation (element_id, type, source_filename, page_number, content, metadata, score)
- Empty sparse results (empty string, whitespace, non-matching query, empty index)
- Clean failure when BM25 is not configured (RuntimeError)
- Input validations (top_k <= 0, query type, bm25_index type, unsupported modality)
- Backward compatibility: dense text and visual retrieval remain unaffected
"""

from typing import List, Optional
import numpy as np
import pytest

from app.embeddings.base import BaseEmbedder
from app.indexing import (
    BM25Index,
    MultimodalFAISSIndex,
    MultimodalSearchResult,
    build_bm25_index,
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
from app.multimodal.embedding import BaseVisualEmbedder
from app.retrieval import (
    MultimodalRetrievalPipeline,
    build_multimodal_pipeline,
)


# ===========================================================================
# Deterministic Test Helpers and Embedders
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
            val = float(hash(t) % 1000) / 1000.0
            vec[idx % 384] = val if val != 0.0 else 0.5
            norm = np.linalg.norm(vec)
            vectors.append(vec / (norm if norm > 0 else 1.0))
        return np.vstack(vectors)


class MockVisualEmbedder(BaseVisualEmbedder):
    """Deterministic 512-d mock visual embedder."""

    @property
    def dimension(self) -> int:
        return 512

    @property
    def model_name(self) -> str:
        return "mock-visual-512"

    def embed_images(self, images: List[object], batch_size: Optional[int] = None) -> np.ndarray:
        if not images:
            return np.empty((0, 512), dtype=np.float32)
        vectors = []
        for idx in range(len(images)):
            vec = np.zeros(512, dtype=np.float32)
            vec[idx % 512] = 1.0
            vectors.append(vec)
        return np.vstack(vectors)


@pytest.fixture
def multimodal_test_document() -> MultimodalDocument:
    """Assembles a rich scientific document containing all 5 modalities."""
    text_el = MultimodalElement(
        element_id="txt-intro",
        element_type=ElementType.TEXT,
        source_filename="transformer_paper.pdf",
        page_number=1,
        content="The dominant sequence transduction models are based on complex recurrent neural networks.",
    )
    caption_el = MultimodalElement(
        element_id="cap-arch",
        element_type=ElementType.CAPTION,
        source_filename="transformer_paper.pdf",
        page_number=2,
        content="Figure 1: The Transformer - model architecture with multi-head attention.",
        related_element_ids=["img-arch"],
        metadata={"label": "Figure 1"},
    )
    table_el = MultimodalElement(
        element_id="tbl-eval",
        element_type=ElementType.TABLE,
        source_filename="transformer_paper.pdf",
        page_number=4,
        content="| Model | BLEU-4 | Training Cost |\n| Transformer-base | 27.3 | 3.3e18 |\n| Transformer-big | 28.4 | 2.3e19 |",
        metadata={"num_rows": 3, "headers": ["Model", "BLEU-4", "Training Cost"]},
    )
    equation_el = MultimodalElement(
        element_id="eq-attn",
        element_type=ElementType.EQUATION,
        source_filename="transformer_paper.pdf",
        page_number=3,
        content=r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V",
        metadata={"is_display": True},
    )
    image_el = MultimodalElement(
        element_id="img-arch",
        element_type=ElementType.IMAGE,
        source_filename="transformer_paper.pdf",
        page_number=2,
        content=b"\x89PNG\r\n\x1a\nFakeDiagramBytes",
        related_element_ids=["cap-arch"],
        metadata={"width": 300, "height": 400},
    )
    return MultimodalDocument(
        elements=[text_el, caption_el, table_el, equation_el, image_el],
        source_filename="transformer_paper.pdf",
    )


@pytest.fixture
def multimodal_pipeline_with_bm25(
    multimodal_test_document: MultimodalDocument,
) -> MultimodalRetrievalPipeline:
    """Creates a MultimodalRetrievalPipeline configured with both FAISS and BM25 indices."""
    embedder = MultimodalEmbedder(
        text_embedder=MockTextEmbedder(),
        visual_embedder=MockVisualEmbedder(),
    )
    faiss_index = build_multimodal_index(multimodal_test_document, embedder=embedder)
    bm25_index = build_bm25_index(multimodal_test_document)

    return build_multimodal_pipeline(
        index=faiss_index,
        text_embedder=MockTextEmbedder(),
        visual_embedder=MockVisualEmbedder(),
        bm25_index=bm25_index,
        default_top_k=3,
    )


# ===========================================================================
# 1. Successful Sparse Retrieval & Delegation Tests
# ===========================================================================


class TestSparseRetrievalDelegation:
    """Verifies that sparse retrieval correctly delegates to BM25Index."""

    def test_successful_sparse_retrieval(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
    ):
        results = multimodal_pipeline_with_bm25.retrieve_sparse("recurrent neural networks")
        assert len(results) >= 1
        top_result = results[0]
        assert isinstance(top_result, MultimodalSearchResult)
        assert top_result.element_id == "txt-intro"
        assert top_result.element_type == ElementType.TEXT
        assert top_result.score > 0.0

    def test_search_sparse_convenience_alias(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
    ):
        res1 = multimodal_pipeline_with_bm25.retrieve_sparse("recurrent neural networks")
        res2 = multimodal_pipeline_with_bm25.search_sparse("recurrent neural networks")
        assert len(res1) == len(res2)
        assert res1[0].element_id == res2[0].element_id
        assert res1[0].score == res2[0].score

    def test_retrieve_modality_sparse_dispatch(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
    ):
        # Dispatch via modality="sparse"
        res_sparse = multimodal_pipeline_with_bm25.retrieve(
            "recurrent neural networks", modality="sparse"
        )
        assert len(res_sparse) >= 1
        assert res_sparse[0].element_id == "txt-intro"

        # Dispatch via modality="bm25"
        res_bm25 = multimodal_pipeline_with_bm25.retrieve(
            "recurrent neural networks", modality="bm25"
        )
        assert len(res_bm25) >= 1
        assert res_bm25[0].element_id == "txt-intro"


# ===========================================================================
# 2. Modality Coverage & IMAGE Exclusion Tests
# ===========================================================================


class TestSparseModalityCoverage:
    """Verifies that TEXT, CAPTION, TABLE, EQUATION can be retrieved via BM25, while IMAGE is excluded."""

    def test_text_retrieval(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
    ):
        results = multimodal_pipeline_with_bm25.retrieve_sparse("transduction models recurrent")
        assert len(results) >= 1
        assert results[0].element_id == "txt-intro"
        assert results[0].element_type == ElementType.TEXT

    def test_caption_retrieval(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
    ):
        results = multimodal_pipeline_with_bm25.retrieve_sparse("Figure 1 architecture diagram")
        assert len(results) >= 1
        assert results[0].element_id == "cap-arch"
        assert results[0].element_type == ElementType.CAPTION

    def test_table_retrieval(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
    ):
        results = multimodal_pipeline_with_bm25.retrieve_sparse("BLEU-4 training cost")
        assert len(results) >= 1
        assert results[0].element_id == "tbl-eval"
        assert results[0].element_type == ElementType.TABLE

    def test_equation_retrieval(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
    ):
        results = multimodal_pipeline_with_bm25.retrieve_sparse("softmax d_k scaling")
        assert len(results) >= 1
        assert results[0].element_id == "eq-attn"
        assert results[0].element_type == ElementType.EQUATION

    def test_image_element_strictly_excluded_from_sparse_results(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
    ):
        # Query for terms associated with diagram/architecture
        results = multimodal_pipeline_with_bm25.retrieve_sparse("architecture diagram", top_k=10)
        # Ensure no element returned has type IMAGE
        for res in results:
            assert res.element_type != ElementType.IMAGE
            assert res.element_id != "img-arch"


# ===========================================================================
# 3. Parameter Propagation and Provenance Preservation
# ===========================================================================


class TestSparseParametersAndProvenance:
    """Verifies top_k propagation and complete provenance preservation."""

    def test_top_k_propagation(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
    ):
        # Query matching multiple documents with "transformer" or "attention"
        res_default = multimodal_pipeline_with_bm25.retrieve_sparse("transformer attention")
        assert len(res_default) <= multimodal_pipeline_with_bm25.default_top_k

        res_1 = multimodal_pipeline_with_bm25.retrieve_sparse("transformer attention", top_k=1)
        assert len(res_1) == 1

        res_2 = multimodal_pipeline_with_bm25.retrieve_sparse("transformer attention", top_k=2)
        assert len(res_2) == 2

    def test_provenance_preservation(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
    ):
        results = multimodal_pipeline_with_bm25.retrieve_sparse("BLEU-4", top_k=1)
        assert len(results) == 1
        hit = results[0]

        assert hit.element_id == "tbl-eval"
        assert hit.element_type == ElementType.TABLE
        assert hit.source_filename == "transformer_paper.pdf"
        assert hit.page_number == 4
        assert hit.metadata["num_rows"] == 3
        assert hit.metadata["headers"] == ["Model", "BLEU-4", "Training Cost"]
        assert "| Model | BLEU-4 | Training Cost |" in hit.content
        assert isinstance(hit.score, float)
        assert hit.score > 0.0


# ===========================================================================
# 4. Empty & Edge Case Behaviors
# ===========================================================================


class TestSparseEmptyAndEdgeCases:
    """Verifies edge cases such as empty query, empty index, and non-matching queries."""

    def test_empty_and_whitespace_query_returns_empty_list(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
    ):
        assert multimodal_pipeline_with_bm25.retrieve_sparse("") == []
        assert multimodal_pipeline_with_bm25.retrieve_sparse("   \t\n ") == []

    def test_non_matching_query_returns_empty_list(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
    ):
        assert (
            multimodal_pipeline_with_bm25.retrieve_sparse("superconducting qubits quantum") == []
        )

    def test_empty_bm25_index_returns_empty_list(self):
        empty_bm25 = BM25Index()
        pipeline = MultimodalRetrievalPipeline(
            index=MultimodalFAISSIndex(),
            bm25_index=empty_bm25,
        )
        assert pipeline.retrieve_sparse("transformer") == []


# ===========================================================================
# 5. Unconfigured BM25 & Validation Tests
# ===========================================================================


class TestUnconfiguredBM25AndValidation:
    """Verifies behavior when BM25 is not configured, and input validation."""

    def test_behavior_when_bm25_is_not_configured(self):
        # Pipeline initialized without BM25
        pipeline = MultimodalRetrievalPipeline(index=MultimodalFAISSIndex())
        assert pipeline.has_bm25 is False
        assert pipeline.bm25_index is None

        # Direct retrieve_sparse call raises RuntimeError
        with pytest.raises(RuntimeError, match="BM25 index is not configured"):
            pipeline.retrieve_sparse("attention")

        # retrieve(modality='sparse') raises RuntimeError
        with pytest.raises(RuntimeError, match="BM25 index is not configured"):
            pipeline.retrieve("attention", modality="sparse")

    def test_invalid_top_k_raises_value_error(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
    ):
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            multimodal_pipeline_with_bm25.retrieve_sparse("attention", top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            multimodal_pipeline_with_bm25.retrieve_sparse("attention", top_k=-3)

    def test_invalid_query_type_raises_type_error(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
    ):
        with pytest.raises(TypeError, match="query must be a string"):
            multimodal_pipeline_with_bm25.retrieve_sparse(12345)

        with pytest.raises(TypeError, match="query must be a string"):
            multimodal_pipeline_with_bm25.retrieve_sparse(["attention"])

    def test_invalid_bm25_index_in_init_raises_type_error(self):
        with pytest.raises(TypeError, match="bm25_index must be an instance of BM25Index"):
            MultimodalRetrievalPipeline(
                index=MultimodalFAISSIndex(),
                bm25_index="not_a_bm25_index",
            )

    def test_unsupported_modality_in_retrieve(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
    ):
        with pytest.raises(ValueError, match="Unsupported modality 'audio'"):
            multimodal_pipeline_with_bm25.retrieve("query", modality="audio")


# ===========================================================================
# 6. Backward Compatibility with Dense Retrieval
# ===========================================================================


class TestDenseRetrievalBackwardCompatibility:
    """Verifies that existing dense text and visual retrieval remain unaffected."""

    def test_dense_text_retrieval_intact(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
    ):
        results = multimodal_pipeline_with_bm25.retrieve_text("attention", top_k=2)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, MultimodalSearchResult)
            assert r.element_type != ElementType.IMAGE

    def test_dense_visual_retrieval_intact(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
    ):
        results = multimodal_pipeline_with_bm25.retrieve_visual(b"\x89PNGfigure", top_k=1)
        assert len(results) == 1
        assert results[0].element_type == ElementType.IMAGE
        assert results[0].element_id == "img-arch"

    def test_retrieve_and_assemble_with_dense_intact(
        self,
        multimodal_pipeline_with_bm25: MultimodalRetrievalPipeline,
        multimodal_test_document: MultimodalDocument,
    ):
        context = multimodal_pipeline_with_bm25.retrieve_and_assemble(
            query="attention",
            modality="text",
            top_k=2,
            document=multimodal_test_document,
        )
        assert len(context) == 2
        assert len(context.citations) == 2
        assert len(context.search_results) == 2
        assert "[1]" in context.formatted_context
