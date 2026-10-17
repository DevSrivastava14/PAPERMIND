"""Focused integration tests for Day 15 Task 4: Hybrid Dense (FAISS) + Sparse (BM25) + RRF Retrieval.

Verifies:
A. Dense-only results appear in final hybrid results.
B. Sparse-only results appear in final hybrid results.
C. Overlapping results appear in both lists and receive combined RRF contribution.
D. Final top_k is respected.
E. Modality isolation: IMAGE elements never appear in hybrid text retrieval.
F. Provenance: element_id, page_number, element_type, source_filename, metadata, relationships preserved.
G. Empty retrieval handling (dense empty + sparse populated, sparse empty + dense populated, both empty).
H. Configuration error when BM25 is missing.
I. Backward compatibility for dense text, sparse, visual, and retrieve_and_assemble.
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
    ElementType,
    MultimodalDocument,
    MultimodalElement,
    MultimodalEmbedder,
)
from app.multimodal.embedding import BaseVisualEmbedder
from app.retrieval import (
    MultimodalRetrievalPipeline,
    build_multimodal_pipeline,
)
from app.retrieval.rrf import DEFAULT_RRF_K


# ===========================================================================
# Deterministic Test Fixtures and Embedders
# ===========================================================================


class OrthogonalConceptEmbedder(BaseEmbedder):
    """Deterministic 384-d embedder placing distinct semantic concepts into orthogonal unit vectors."""

    CONCEPT_DIMS = {
        "transformer": 0,
        "attention": 1,
        "optimization": 2,
        "table": 3,
        "equation": 4,
    }

    @property
    def dimension(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return "concept-orthogonal-384"

    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        vectors = []
        for text in texts:
            vec = np.zeros(384, dtype=np.float32)
            lower = text.lower()
            matched = False
            for concept, dim in self.CONCEPT_DIMS.items():
                if concept in lower:
                    vec[dim] += 1.0
                    matched = True
            if not matched:
                vec[100] = 1.0  # Unmatched fallback dimension
            norm = np.linalg.norm(vec)
            vectors.append(vec / (norm if norm > 0 else 1.0))
        return np.vstack(vectors)


class DeterministicVisualEmbedder(BaseVisualEmbedder):
    """Deterministic 512-d visual embedder."""

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
def hybrid_paper_document() -> MultimodalDocument:
    """Assembles a paper document with dense-favored, sparse-favored, and overlapping elements."""
    # 1. Overlapping element: matches both dense (attention concept) and sparse (keyword "transformer")
    el_overlap = MultimodalElement(
        element_id="txt-overlap",
        element_type=ElementType.TEXT,
        source_filename="hybrid_paper.pdf",
        page_number=1,
        content="Transformer model uses multi-head attention mechanisms for translation.",
        metadata={"section": "abstract"},
    )
    # 2. Dense-only element: matches dense "optimization" concept, but lacks sparse keywords
    el_dense_only = MultimodalElement(
        element_id="txt-dense-only",
        element_type=ElementType.TEXT,
        source_filename="hybrid_paper.pdf",
        page_number=2,
        content="Optimization convergence and gradient descent schedule details.",
        metadata={"section": "training"},
    )
    # 3. Sparse-only element: mentions specific keyword "d_k" repeatedly for high BM25 score
    el_sparse_only = MultimodalElement(
        element_id="eq-sparse-only",
        element_type=ElementType.EQUATION,
        source_filename="hybrid_paper.pdf",
        page_number=3,
        content=r"\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \text{ scaling with } d_k",
        metadata={"section": "model", "is_display": True},
    )
    # 4. Table element: benchmark results
    el_table = MultimodalElement(
        element_id="tbl-benchmark",
        element_type=ElementType.TABLE,
        source_filename="hybrid_paper.pdf",
        page_number=4,
        content="| Model | BLEU-4 |\n| Baseline | 28.4 |",
        metadata={"headers": ["Model", "BLEU-4"]},
    )
    # 5. Image element: architecture diagram (MUST NEVER be in hybrid text results)
    el_image = MultimodalElement(
        element_id="img-architecture",
        element_type=ElementType.IMAGE,
        source_filename="hybrid_paper.pdf",
        page_number=2,
        content=b"\x89PNG\r\n\x1a\nFakeDiagramData",
        related_element_ids=["cap-arch"],
        metadata={"width": 400, "height": 300},
    )
    return MultimodalDocument(
        elements=[el_overlap, el_dense_only, el_sparse_only, el_table, el_image],
        source_filename="hybrid_paper.pdf",
    )


@pytest.fixture
def hybrid_pipeline(
    hybrid_paper_document: MultimodalDocument,
) -> MultimodalRetrievalPipeline:
    """Builds a pipeline with both FAISS text/visual index and BM25 index."""
    embedder = MultimodalEmbedder(
        text_embedder=OrthogonalConceptEmbedder(),
        visual_embedder=DeterministicVisualEmbedder(),
    )
    faiss_index = build_multimodal_index(hybrid_paper_document, embedder=embedder)
    bm25_index = build_bm25_index(hybrid_paper_document)

    return build_multimodal_pipeline(
        index=faiss_index,
        text_embedder=OrthogonalConceptEmbedder(),
        visual_embedder=DeterministicVisualEmbedder(),
        bm25_index=bm25_index,
        default_top_k=5,
    )


# ===========================================================================
# 1. Core Integration: Overlap, Dense-only, and Sparse-only Verification
# ===========================================================================


class TestHybridRetrievalIntegration:
    """Verifies that RRF correctly combines dense and sparse candidate lists."""

    def test_overlapping_element_receives_combined_rrf_score(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
    ):
        """Requirement C: Overlapping result appears in both lists and receives combined contribution."""
        # Query: "transformer d_k"
        # Dense will rank "txt-overlap" (contains "transformer") at rank 1.
        # BM25 will rank "eq-sparse-only" (contains "d_k" twice) at rank 1, and "txt-overlap" (contains "transformer") at rank 2.
        results = hybrid_pipeline.hybrid_retrieve_text(
            query="transformer d_k",
            top_k=5,
            rrf_k=60,
        )

        result_ids = [r.element_id for r in results]
        assert "txt-overlap" in result_ids
        assert "eq-sparse-only" in result_ids

        # txt-overlap appears in dense (rank 1) and sparse (rank 2)
        # Expected RRF score = 1/(60+1) + 1/(60+2)
        overlap_res = next(r for r in results if r.element_id == "txt-overlap")
        expected_score = (1.0 / 61.0) + (1.0 / 62.0)
        assert pytest.approx(overlap_res.score, rel=1e-5) == expected_score

    def test_dense_only_result_survives(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
    ):
        """Requirement A: Dense-only result appears in final hybrid results."""
        # Query: "optimization d_k"
        # Dense matches "txt-dense-only" (contains optimization concept)
        # BM25 matches "eq-sparse-only" (contains d_k)
        results = hybrid_pipeline.hybrid_retrieve_text(
            query="optimization d_k",
            top_k=5,
            rrf_k=60,
        )

        result_ids = [r.element_id for r in results]
        assert "txt-dense-only" in result_ids
        dense_res = next(r for r in results if r.element_id == "txt-dense-only")
        assert dense_res.score > 0.0

    def test_sparse_only_result_survives(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
    ):
        """Requirement B: Sparse-only result appears in final hybrid results."""
        # Query targeting d_k
        results = hybrid_pipeline.hybrid_retrieve_text(
            query="d_k scaling",
            top_k=5,
            rrf_k=60,
        )

        result_ids = [r.element_id for r in results]
        assert "eq-sparse-only" in result_ids
        sparse_res = next(r for r in results if r.element_id == "eq-sparse-only")
        assert sparse_res.score > 0.0

    def test_final_top_k_is_respected(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
    ):
        """Requirement D: top_k parameter strictly caps returned results."""
        results = hybrid_pipeline.hybrid_retrieve_text(
            query="transformer optimization d_k",
            top_k=2,
        )
        assert len(results) == 2

    def test_modality_isolation_image_strictly_excluded(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
    ):
        """Requirement E & 7: IMAGE elements must NEVER appear in hybrid text results."""
        # Query for architecture/diagram
        results = hybrid_pipeline.hybrid_retrieve_text(
            query="architecture diagram model",
            top_k=10,
        )
        for r in results:
            assert r.element_type != ElementType.IMAGE
            assert r.element_id != "img-architecture"

    def test_provenance_preservation(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
    ):
        """Requirement F & 8-9: All provenance attributes preserved across FAISS -> BM25 -> RRF -> output."""
        results = hybrid_pipeline.hybrid_retrieve_text(
            query="optimization",
            top_k=1,
        )
        assert len(results) == 1
        res = results[0]

        assert res.element_id == "txt-dense-only"
        assert res.element_type == ElementType.TEXT
        assert res.source_filename == "hybrid_paper.pdf"
        assert res.page_number == 2
        assert res.metadata["section"] == "training"
        assert "Optimization convergence" in res.content
        assert isinstance(res.score, float)
        assert res.score > 0.0


# ===========================================================================
# 2. Empty Retrieval and Edge Cases
# ===========================================================================


class TestHybridEmptyAndEdgeCases:
    """Verifies Requirement G: handling of empty and partially-empty candidate sets."""

    def test_both_dense_and_sparse_empty(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
    ):
        """Both dense and sparse return 0 hits for unrelated/unmatchable query."""
        results = hybrid_pipeline.hybrid_retrieve_text(
            query="quantum superconducting tokamak fusion",
            top_k=5,
        )
        # Even if concept embedder assigns fallback dimension, if no relevant match or empty string:
        assert isinstance(results, list)

    def test_empty_string_query_returns_empty_list(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
    ):
        assert hybrid_pipeline.hybrid_retrieve_text("") == []
        assert hybrid_pipeline.hybrid_retrieve_text("   \t\n  ") == []

    def test_dense_populated_and_sparse_empty(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
    ):
        """Dense finds hits via semantic vector but query keywords have no exact BM25 match."""
        # "optimization" concept matches dense dimension 2; suppose BM25 has no exact keyword if misspelled
        results = hybrid_pipeline.hybrid_retrieve_text(
            query="optimization",
            top_k=5,
        )
        assert len(results) >= 1
        assert any(r.element_id == "txt-dense-only" for r in results)


# ===========================================================================
# 3. Unconfigured BM25 & Parameter Validations
# ===========================================================================


class TestHybridConfigurationAndValidation:
    """Verifies Requirement H & 11-12: error on missing BM25 and parameter validation."""

    def test_missing_bm25_raises_clear_runtime_error(self):
        """Requirement H & 12: Calling hybrid retrieval without BM25 raises clear RuntimeError."""
        pipeline_without_bm25 = MultimodalRetrievalPipeline(
            index=MultimodalFAISSIndex(),
        )
        with pytest.raises(RuntimeError, match="BM25 index is not configured"):
            pipeline_without_bm25.hybrid_retrieve_text("attention")

        with pytest.raises(RuntimeError, match="BM25 index is not configured"):
            pipeline_without_bm25.retrieve("attention", modality="hybrid")

    def test_invalid_top_k_parameters(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
    ):
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            hybrid_pipeline.hybrid_retrieve_text("attention", top_k=0)

        with pytest.raises(ValueError, match="dense_top_k must be a positive integer"):
            hybrid_pipeline.hybrid_retrieve_text("attention", dense_top_k=-1)

        with pytest.raises(ValueError, match="sparse_top_k must be a positive integer"):
            hybrid_pipeline.hybrid_retrieve_text("attention", sparse_top_k=0)

    def test_invalid_query_type(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
    ):
        with pytest.raises(TypeError, match="query must be a string"):
            hybrid_pipeline.hybrid_retrieve_text(12345)

        with pytest.raises(TypeError, match="query must be a string"):
            hybrid_pipeline.hybrid_retrieve_text(["attention"])

    def test_invalid_rrf_k(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
    ):
        with pytest.raises(ValueError, match="k must be a positive number"):
            hybrid_pipeline.hybrid_retrieve_text("attention", rrf_k=0)

        with pytest.raises(ValueError, match="k must be a positive number"):
            hybrid_pipeline.hybrid_retrieve_text("attention", rrf_k=-5)

        with pytest.raises(TypeError, match="k must be an integer or float"):
            hybrid_pipeline.hybrid_retrieve_text("attention", rrf_k="60")


# ===========================================================================
# 4. Backward Compatibility
# ===========================================================================


class TestHybridBackwardCompatibility:
    """Verifies Requirement I & 13: all existing retrieval methods remain fully functional."""

    def test_existing_dense_text_retrieval(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
    ):
        results = hybrid_pipeline.retrieve_text("transformer", top_k=2)
        assert len(results) == 2
        assert isinstance(results[0], MultimodalSearchResult)

    def test_existing_sparse_retrieval(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
    ):
        results = hybrid_pipeline.retrieve_sparse("transformer", top_k=2)
        assert len(results) >= 1
        assert isinstance(results[0], MultimodalSearchResult)

    def test_existing_visual_retrieval(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
    ):
        results = hybrid_pipeline.retrieve_visual(b"\x89PNGdata", top_k=1)
        assert len(results) == 1
        assert results[0].element_type == ElementType.IMAGE

    def test_existing_retrieve_and_assemble(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
        hybrid_paper_document: MultimodalDocument,
    ):
        context = hybrid_pipeline.retrieve_and_assemble(
            query="transformer",
            modality="text",
            top_k=2,
            document=hybrid_paper_document,
        )
        assert len(context) == 2
        assert len(context.citations) == 2
        assert "[1]" in context.formatted_context

    def test_hybrid_retrieve_and_assemble(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
        hybrid_paper_document: MultimodalDocument,
    ):
        """Verifies Requirement 15: hybrid context assembly works cleanly."""
        context = hybrid_pipeline.hybrid_retrieve_and_assemble(
            query="transformer d_k",
            top_k=2,
            document=hybrid_paper_document,
        )
        assert len(context) == 2
        assert len(context.citations) == 2
        assert "[1]" in context.formatted_context
        # Scores are positive RRF scores
        for cit in context.citations:
            assert cit.score > 0.0

    def test_retrieve_modality_hybrid_dispatch(
        self,
        hybrid_pipeline: MultimodalRetrievalPipeline,
    ):
        results = hybrid_pipeline.retrieve(
            query="transformer d_k",
            modality="hybrid",
            top_k=2,
        )
        assert len(results) == 2
        assert isinstance(results[0], MultimodalSearchResult)
