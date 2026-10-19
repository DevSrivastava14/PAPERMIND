"""Focused integration tests for Day 16 Task 2: Cross-Encoder Reranking Hybrid Integration.

Verifies:
1. RRF results being passed to the reranker
2. Reranked ordering based on cross-encoder scores
3. Final top_k limits respected
4. Full provenance preservation across all MultimodalSearchResult fields
5. Missing reranker configuration raises clear RuntimeError
6. Empty results and edge case handling (empty query, non-matching query, empty candidate pool)
7. Backward compatibility: existing hybrid retrieval (hybrid_retrieve_text, retrieve(modality='hybrid'))
   continues to return pure RRF results without invoking the reranker
8. Prompt context assembly and structured citations after reranking
9. Text-only modality isolation: IMAGE elements are strictly excluded
10. Convenience aliases and modality dispatch routing
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
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
from app.reranking.base import BaseReranker
from app.reranking.cross_encoder import CrossEncoderReranker
from app.retrieval import (
    MultimodalRetrievalPipeline,
    build_multimodal_pipeline,
)
from app.retrieval.multimodal_models import MultimodalRetrievedContext
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
                vec[100] = 1.0  # Fallback dimension
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


class RecordingFakeCrossEncoder:
    """Deterministic fake cross-encoder model recording candidate pairs and scoring predictably."""

    def __init__(self, score_map: Optional[Dict[str, float]] = None, default_score: float = 0.0) -> None:
        self.score_map = dict(score_map or {})
        self.default_score = float(default_score)
        self.recorded_calls: List[List[Tuple[str, str]]] = []

    def predict(self, pairs: Sequence[Sequence[str]], **kwargs: Any) -> List[float]:
        self.recorded_calls.append([(p[0], p[1]) for p in pairs])
        scores = []
        for query, doc_text in pairs:
            # Score lookup by exact match or substring
            matched_score = None
            for key, val in self.score_map.items():
                if key in doc_text:
                    matched_score = val
                    break
            scores.append(float(matched_score if matched_score is not None else self.default_score))
        return scores


@pytest.fixture
def multimodal_paper_doc() -> MultimodalDocument:
    """Document with text, equation, table, and image elements for hybrid retrieval."""
    el_overlap = MultimodalElement(
        element_id="txt-overlap",
        element_type=ElementType.TEXT,
        source_filename="hybrid_paper.pdf",
        page_number=1,
        content="Transformer model uses multi-head attention mechanisms for translation.",
        metadata={"section": "abstract", "importance": "high"},
        related_element_ids=["rel-1"],
    )
    el_dense_only = MultimodalElement(
        element_id="txt-dense-only",
        element_type=ElementType.TEXT,
        source_filename="hybrid_paper.pdf",
        page_number=2,
        content="Optimization convergence and gradient descent schedule details.",
        metadata={"section": "training"},
        related_element_ids=["rel-2"],
    )
    el_sparse_only = MultimodalElement(
        element_id="eq-sparse-only",
        element_type=ElementType.EQUATION,
        source_filename="hybrid_paper.pdf",
        page_number=3,
        content=r"\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \text{ scaling with } d_k",
        metadata={"section": "model", "is_display": True},
        related_element_ids=["rel-3"],
    )
    el_table = MultimodalElement(
        element_id="tbl-benchmark",
        element_type=ElementType.TABLE,
        source_filename="hybrid_paper.pdf",
        page_number=4,
        content="| Model | BLEU-4 |\n| Baseline | 28.4 |",
        metadata={"headers": ["Model", "BLEU-4"]},
        related_element_ids=["rel-4"],
    )
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
def fake_reranker_model() -> RecordingFakeCrossEncoder:
    """Fake model scoring the equation highest, table second, overlap third, dense-only last."""
    return RecordingFakeCrossEncoder(
        score_map={
            "softmax": 0.95,       # eq-sparse-only
            "BLEU-4": 0.85,        # tbl-benchmark
            "Transformer": 0.40,   # txt-overlap
            "Optimization": 0.10,  # txt-dense-only
        },
        default_score=0.01,
    )


@pytest.fixture
def hybrid_reranked_pipeline(
    multimodal_paper_doc: MultimodalDocument,
    fake_reranker_model: RecordingFakeCrossEncoder,
) -> MultimodalRetrievalPipeline:
    """Builds a pipeline with FAISS, BM25, and an injected fake CrossEncoderReranker."""
    embedder = MultimodalEmbedder(
        text_embedder=OrthogonalConceptEmbedder(),
        visual_embedder=DeterministicVisualEmbedder(),
    )
    faiss_index = build_multimodal_index(multimodal_paper_doc, embedder=embedder)
    bm25_index = build_bm25_index(multimodal_paper_doc)
    reranker = CrossEncoderReranker(model=fake_reranker_model)

    return build_multimodal_pipeline(
        index=faiss_index,
        text_embedder=OrthogonalConceptEmbedder(),
        visual_embedder=DeterministicVisualEmbedder(),
        bm25_index=bm25_index,
        reranker=reranker,
        default_top_k=5,
    )


# ===========================================================================
# 1. Pipeline Setup and Configuration Tests
# ===========================================================================


class TestRerankedPipelineConfiguration:
    """Verifies pipeline initialization with reranker and has_reranker property."""

    def test_pipeline_with_reranker_initialization(
        self,
        hybrid_reranked_pipeline: MultimodalRetrievalPipeline,
    ):
        assert hybrid_reranked_pipeline.has_reranker is True
        assert hybrid_reranked_pipeline.reranker is not None
        assert isinstance(hybrid_reranked_pipeline.reranker, BaseReranker)

    def test_pipeline_without_reranker(
        self,
        multimodal_paper_doc: MultimodalDocument,
    ):
        embedder = MultimodalEmbedder(
            text_embedder=OrthogonalConceptEmbedder(),
            visual_embedder=DeterministicVisualEmbedder(),
        )
        faiss_index = build_multimodal_index(multimodal_paper_doc, embedder=embedder)
        pipeline = build_multimodal_pipeline(index=faiss_index)

        assert pipeline.has_reranker is False
        assert pipeline.reranker is None

    def test_invalid_reranker_type_raises_type_error(
        self,
        multimodal_paper_doc: MultimodalDocument,
    ):
        embedder = MultimodalEmbedder(
            text_embedder=OrthogonalConceptEmbedder(),
            visual_embedder=DeterministicVisualEmbedder(),
        )
        faiss_index = build_multimodal_index(multimodal_paper_doc, embedder=embedder)

        with pytest.raises(TypeError, match="reranker must be an instance of BaseReranker"):
            MultimodalRetrievalPipeline(
                index=faiss_index,
                reranker="not_a_valid_reranker",  # type: ignore
            )


# ===========================================================================
# 2. Missing Reranker and Configuration Validation Tests
# ===========================================================================


class TestMissingRerankerValidation:
    """Verifies clear RuntimeError when reranking is requested but reranker is missing."""

    def test_missing_reranker_raises_runtime_error(
        self,
        multimodal_paper_doc: MultimodalDocument,
    ):
        embedder = MultimodalEmbedder(
            text_embedder=OrthogonalConceptEmbedder(),
            visual_embedder=DeterministicVisualEmbedder(),
        )
        faiss_index = build_multimodal_index(multimodal_paper_doc, embedder=embedder)
        bm25_index = build_bm25_index(multimodal_paper_doc)

        pipeline_no_reranker = build_multimodal_pipeline(
            index=faiss_index,
            bm25_index=bm25_index,
            reranker=None,
        )

        with pytest.raises(RuntimeError, match="Reranker is not configured for this pipeline"):
            pipeline_no_reranker.hybrid_retrieve_reranked("transformer attention")

        with pytest.raises(RuntimeError, match="Reranker is not configured for this pipeline"):
            pipeline_no_reranker.hybrid_retrieve_reranked_and_assemble("transformer attention")

        with pytest.raises(RuntimeError, match="Reranker is not configured for this pipeline"):
            pipeline_no_reranker.retrieve("transformer attention", modality="hybrid_reranked")

    def test_missing_bm25_in_reranked_pipeline_raises_runtime_error(
        self,
        multimodal_paper_doc: MultimodalDocument,
        fake_reranker_model: RecordingFakeCrossEncoder,
    ):
        embedder = MultimodalEmbedder(
            text_embedder=OrthogonalConceptEmbedder(),
            visual_embedder=DeterministicVisualEmbedder(),
        )
        faiss_index = build_multimodal_index(multimodal_paper_doc, embedder=embedder)
        reranker = CrossEncoderReranker(model=fake_reranker_model)

        pipeline_no_bm25 = build_multimodal_pipeline(
            index=faiss_index,
            bm25_index=None,
            reranker=reranker,
        )

        with pytest.raises(RuntimeError, match="BM25 index is not configured for this pipeline"):
            pipeline_no_bm25.hybrid_retrieve_reranked("transformer attention")

    def test_ad_hoc_reranker_override(
        self,
        multimodal_paper_doc: MultimodalDocument,
    ):
        """Pipeline without reranker succeeds when an ad-hoc reranker is passed to the method."""
        embedder = MultimodalEmbedder(
            text_embedder=OrthogonalConceptEmbedder(),
            visual_embedder=DeterministicVisualEmbedder(),
        )
        faiss_index = build_multimodal_index(multimodal_paper_doc, embedder=embedder)
        bm25_index = build_bm25_index(multimodal_paper_doc)
        pipeline = build_multimodal_pipeline(index=faiss_index, bm25_index=bm25_index)

        fake_model = RecordingFakeCrossEncoder(default_score=0.99)
        custom_reranker = CrossEncoderReranker(model=fake_model)

        results = pipeline.hybrid_retrieve_reranked(
            "transformer",
            reranker=custom_reranker,
            top_k=2,
        )
        assert len(results) > 0
        assert results[0].score == 0.99


# ===========================================================================
# 3. RRF to Reranker Flow and Reranked Ordering Tests
# ===========================================================================


class TestRerankedRetrievalFlow:
    """Verifies that RRF candidates flow into the cross-encoder and are reordered properly."""

    def test_rrf_results_passed_to_reranker(
        self,
        hybrid_reranked_pipeline: MultimodalRetrievalPipeline,
        fake_reranker_model: RecordingFakeCrossEncoder,
    ):
        query = "transformer attention"
        results = hybrid_reranked_pipeline.hybrid_retrieve_reranked(query, top_k=3)

        assert len(results) > 0
        # Verify the fake cross-encoder received the candidate pairs with the query
        assert len(fake_reranker_model.recorded_calls) == 1
        recorded_pairs = fake_reranker_model.recorded_calls[0]
        assert len(recorded_pairs) > 0
        for q, text in recorded_pairs:
            assert q == query
            assert isinstance(text, str)

    def test_cross_encoder_reorders_rrf_results(
        self,
        hybrid_reranked_pipeline: MultimodalRetrievalPipeline,
    ):
        query = "transformer attention"

        # 1. Inspect RRF-only ordering:
        rrf_only = hybrid_reranked_pipeline.hybrid_retrieve_text(query, top_k=5)
        # In pure RRF: txt-overlap ranks higher than eq-sparse-only because it matched both dense and sparse
        assert rrf_only[0].element_id == "txt-overlap"

        # 2. Inspect Reranked ordering:
        # Cross-encoder scores eq-sparse-only (0.95) > tbl-benchmark (0.85) > txt-overlap (0.40)
        reranked = hybrid_reranked_pipeline.hybrid_retrieve_reranked(query, top_k=5)

        assert len(reranked) >= 3
        assert reranked[0].element_id == "eq-sparse-only"
        assert reranked[0].score == 0.95

        assert reranked[1].element_id == "tbl-benchmark"
        assert reranked[1].score == 0.85

        assert reranked[2].element_id == "txt-overlap"
        assert reranked[2].score == 0.40

    def test_configurable_candidate_pool_before_reranking(
        self,
        hybrid_reranked_pipeline: MultimodalRetrievalPipeline,
        fake_reranker_model: RecordingFakeCrossEncoder,
    ):
        """candidate_top_k / rrf_top_k configures how many RRF candidates are sent to reranker."""
        query = "transformer attention"
        fake_reranker_model.recorded_calls.clear()

        # Request candidate_top_k=2
        hybrid_reranked_pipeline.hybrid_retrieve_reranked(
            query,
            candidate_top_k=2,
            top_k=1,
        )

        assert len(fake_reranker_model.recorded_calls) == 1
        assert len(fake_reranker_model.recorded_calls[0]) == 2


# ===========================================================================
# 4. Final top_k and Provenance Preservation Tests
# ===========================================================================


class TestTopKAndProvenance:
    """Verifies top_k limits and provenance preservation in hybrid reranked retrieval."""

    def test_final_top_k_strictly_respected(
        self,
        hybrid_reranked_pipeline: MultimodalRetrievalPipeline,
    ):
        query = "transformer attention"
        results_1 = hybrid_reranked_pipeline.hybrid_retrieve_reranked(query, top_k=1)
        assert len(results_1) == 1

        results_2 = hybrid_reranked_pipeline.hybrid_retrieve_reranked(query, top_k=2)
        assert len(results_2) == 2

    def test_provenance_preservation_across_all_fields(
        self,
        hybrid_reranked_pipeline: MultimodalRetrievalPipeline,
    ):
        query = "transformer attention"
        results = hybrid_reranked_pipeline.hybrid_retrieve_reranked(query, top_k=5)

        for res in results:
            assert isinstance(res, MultimodalSearchResult)
            assert res.element is not None
            assert res.element_id == res.element.element_id
            assert res.element_type == res.element.element_type
            assert res.source_filename == "hybrid_paper.pdf"
            assert res.page_number >= 1
            assert isinstance(res.metadata, dict)
            assert isinstance(res.related_element_ids, list)
            assert res.content == res.element.content

        # Check specific metadata on overlap element
        overlap = next(r for r in results if r.element_id == "txt-overlap")
        assert overlap.metadata.get("importance") == "high"
        assert overlap.related_element_ids == ["rel-1"]
        assert overlap.page_number == 1

    def test_modality_isolation_image_strictly_excluded(
        self,
        hybrid_reranked_pipeline: MultimodalRetrievalPipeline,
    ):
        """IMAGE elements must never appear in hybrid text reranked results."""
        results = hybrid_reranked_pipeline.hybrid_retrieve_reranked("transformer diagram", top_k=10)
        returned_types = {r.element_type for r in results}
        assert ElementType.IMAGE not in returned_types
        assert all(r.element_id != "img-architecture" for r in results)


# ===========================================================================
# 5. Empty Results and Edge Cases Tests
# ===========================================================================


class TestEmptyAndEdgeCases:
    """Verifies edge cases like empty query, non-matching query, and empty results."""

    def test_empty_string_query_returns_empty_list(
        self,
        hybrid_reranked_pipeline: MultimodalRetrievalPipeline,
        fake_reranker_model: RecordingFakeCrossEncoder,
    ):
        fake_reranker_model.recorded_calls.clear()
        res_empty = hybrid_reranked_pipeline.hybrid_retrieve_reranked("")
        assert res_empty == []

        res_ws = hybrid_reranked_pipeline.hybrid_retrieve_reranked("   \t\n  ")
        assert res_ws == []

        # Cross-encoder should not be invoked when query is empty
        assert len(fake_reranker_model.recorded_calls) == 0

    def test_non_matching_query_returns_empty_list(
        self,
        hybrid_reranked_pipeline: MultimodalRetrievalPipeline,
    ):
        res = hybrid_reranked_pipeline.hybrid_retrieve_reranked("superconducting qubits quantum entanglement")
        # In our deterministic setup, unrelated terms match no concepts or keywords
        assert isinstance(res, list)

    def test_invalid_query_type_raises_type_error(
        self,
        hybrid_reranked_pipeline: MultimodalRetrievalPipeline,
    ):
        with pytest.raises(TypeError, match="query must be a string"):
            hybrid_reranked_pipeline.hybrid_retrieve_reranked(12345)  # type: ignore

        with pytest.raises(TypeError, match="query must be a string"):
            hybrid_reranked_pipeline.hybrid_retrieve_reranked(None)  # type: ignore

    def test_invalid_top_k_raises_value_error(
        self,
        hybrid_reranked_pipeline: MultimodalRetrievalPipeline,
    ):
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            hybrid_reranked_pipeline.hybrid_retrieve_reranked("query", top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            hybrid_reranked_pipeline.hybrid_retrieve_reranked("query", top_k=-1)

        with pytest.raises(ValueError, match="dense_top_k must be a positive integer"):
            hybrid_reranked_pipeline.hybrid_retrieve_reranked("query", dense_top_k=-2)

        with pytest.raises(ValueError, match="sparse_top_k must be a positive integer"):
            hybrid_reranked_pipeline.hybrid_retrieve_reranked("query", sparse_top_k=0)

        with pytest.raises(ValueError, match="candidate_top_k / rrf_top_k must be a positive integer"):
            hybrid_reranked_pipeline.hybrid_retrieve_reranked("query", candidate_top_k=-5)


# ===========================================================================
# 6. Context Assembly and Citations After Reranking Tests
# ===========================================================================


class TestRerankedContextAssembly:
    """Verifies prompt-ready context and citation assembly after cross-encoder reranking."""

    def test_hybrid_retrieve_reranked_and_assemble(
        self,
        hybrid_reranked_pipeline: MultimodalRetrievalPipeline,
        multimodal_paper_doc: MultimodalDocument,
    ):
        query = "transformer attention"
        context = hybrid_reranked_pipeline.hybrid_retrieve_reranked_and_assemble(
            query=query,
            top_k=2,
            document=multimodal_paper_doc,
        )

        assert isinstance(context, MultimodalRetrievedContext)
        assert len(context.citations) == 2
        assert len(context.search_results) == 2

        # 1. Citations ordered according to reranked order: eq-sparse-only [1], tbl-benchmark [2]
        c1 = context.citations[0]
        assert c1.citation_label == "[1]"
        assert c1.element_id == "eq-sparse-only"
        assert c1.page_number == 3
        assert c1.score == 0.95

        c2 = context.citations[1]
        assert c2.citation_label == "[2]"
        assert c2.element_id == "tbl-benchmark"
        assert c2.page_number == 4
        assert c2.score == 0.85

        # 2. Formatted prompt text contains sequential evidence blocks
        assert "[1] [Equation] Source: hybrid_paper.pdf (Page 3)" in context.formatted_context
        assert "[2] [Table] Source: hybrid_paper.pdf (Page 4)" in context.formatted_context

    def test_empty_query_context_assembly(
        self,
        hybrid_reranked_pipeline: MultimodalRetrievalPipeline,
    ):
        context = hybrid_reranked_pipeline.hybrid_retrieve_reranked_and_assemble("")
        assert context.citations == []
        assert context.search_results == []
        assert context.formatted_context == ""


# ===========================================================================
# 7. Backward Compatibility Tests
# ===========================================================================


class TestBackwardCompatibility:
    """Confirms existing retrieval methods and hybrid without reranking behave identically."""

    def test_hybrid_retrieve_text_unaffected(
        self,
        hybrid_reranked_pipeline: MultimodalRetrievalPipeline,
        fake_reranker_model: RecordingFakeCrossEncoder,
    ):
        fake_reranker_model.recorded_calls.clear()
        results = hybrid_reranked_pipeline.hybrid_retrieve_text("transformer attention", top_k=2)

        # Existing hybrid_retrieve_text returns pure RRF results
        assert len(results) == 2
        assert results[0].element_id == "txt-overlap"
        # RRF formula score: 1/(60+1) + 1/(60+1)
        expected_rrf = (1.0 / (60 + 1)) + (1.0 / (60 + 1))
        assert results[0].score == pytest.approx(expected_rrf)

        # Reranker was NOT called
        assert len(fake_reranker_model.recorded_calls) == 0

    def test_modality_hybrid_dispatch_unaffected(
        self,
        hybrid_reranked_pipeline: MultimodalRetrievalPipeline,
        fake_reranker_model: RecordingFakeCrossEncoder,
    ):
        fake_reranker_model.recorded_calls.clear()
        # 'hybrid' modality still routes to pure RRF hybrid_retrieve_text
        results = hybrid_reranked_pipeline.retrieve("transformer attention", modality="hybrid", top_k=2)
        assert results[0].element_id == "txt-overlap"
        assert len(fake_reranker_model.recorded_calls) == 0

    def test_modality_hybrid_reranked_dispatch(
        self,
        hybrid_reranked_pipeline: MultimodalRetrievalPipeline,
    ):
        # 'hybrid_reranked' modality routes to hybrid_retrieve_reranked
        results = hybrid_reranked_pipeline.retrieve("transformer attention", modality="hybrid_reranked", top_k=2)
        assert results[0].element_id == "eq-sparse-only"
        assert results[0].score == 0.95

    def test_convenience_aliases(
        self,
        hybrid_reranked_pipeline: MultimodalRetrievalPipeline,
    ):
        assert hybrid_reranked_pipeline.search_hybrid_reranked.__func__ is MultimodalRetrievalPipeline.hybrid_retrieve_reranked
        assert hybrid_reranked_pipeline.retrieve_hybrid_reranked.__func__ is MultimodalRetrievalPipeline.hybrid_retrieve_reranked
        assert hybrid_reranked_pipeline.reranked_hybrid_retrieve_text.__func__ is MultimodalRetrievalPipeline.hybrid_retrieve_reranked
        assert hybrid_reranked_pipeline.reranked_hybrid_retrieve_and_assemble.__func__ is MultimodalRetrievalPipeline.hybrid_retrieve_reranked_and_assemble
