"""Focused unit tests for Day 16 Task 1: Cross-Encoder Reranking Foundation.

Verifies:
- Abstract BaseReranker interface and contract compliance
- CrossEncoderReranker implementation using injected deterministic fake/mock model
- Candidates reordered by cross-encoder relevance score descending
- Full provenance preservation across all MultimodalSearchResult fields (element, id, type,
  source_filename, page_number, related_element_ids, content, metadata)
- Only score/ranking changes; original attributes intact
- Deterministic tie-breaking by element_id for identical scores
- top_k parameter slicing (top_k=None, top_k <= len, top_k > len)
- Empty candidates handling (returns empty list without invoking model)
- Input validations (invalid query, empty/whitespace query, invalid top_k, invalid candidates, invalid batch_size)
- Model score-count mismatch handling (raises ScoreCountMismatchError / RuntimeError / ValueError)
- Score normalization across various model output formats (1D array, 2D array, lists, scalars)
- Safe handling of multimodal elements (TEXT, CAPTION, TABLE, EQUATION, IMAGE)
- Custom text_extractor injection support
- Convenience functions and import availability from app.reranking and app.retrieval.reranker
- Zero model weight downloads or internet access during testing
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pytest

from app.indexing.multimodal_models import MultimodalSearchResult
from app.multimodal.representation import ElementType, MultimodalElement
from app.reranking.base import BaseReranker
from app.reranking.cross_encoder import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CROSS_ENCODER_MODEL,
    CrossEncoderReranker,
    ScoreCountMismatchError,
    default_candidate_text_extractor,
    rerank_cross_encoder,
)
from app.retrieval.reranker import (
    BaseReranker as RetBaseReranker,
    CrossEncoderReranker as RetCrossEncoderReranker,
)


# ===========================================================================
# Deterministic Fake / Mock Cross-Encoder
# ===========================================================================


class DeterministicFakeCrossEncoder:
    """Predictable in-memory fake cross-encoder model for isolated testing."""

    def __init__(
        self,
        score_map: Optional[Dict[str, float]] = None,
        default_score: float = 0.0,
        fixed_scores: Optional[Sequence[float]] = None,
        output_format: str = "list",
    ) -> None:
        self.score_map = dict(score_map or {})
        self.default_score = float(default_score)
        self.fixed_scores = list(fixed_scores) if fixed_scores is not None else None
        self.output_format = output_format
        self.call_history: List[List[Tuple[str, str]]] = []

    def predict(
        self,
        sentences: Sequence[Sequence[str]],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> Any:
        self.call_history.append([(pair[0], pair[1]) for pair in sentences])

        if self.fixed_scores is not None:
            raw = list(self.fixed_scores)
        else:
            raw = []
            for query, doc_text in sentences:
                score = self.score_map.get(doc_text, self.default_score)
                raw.append(float(score))

        if self.output_format == "list":
            return raw
        elif self.output_format == "numpy_1d":
            return np.array(raw, dtype=np.float32)
        elif self.output_format == "numpy_2d_col":
            return np.array(raw, dtype=np.float32).reshape(-1, 1)
        elif self.output_format == "numpy_2d_logits":
            # [neg_score, pos_score]
            return np.column_stack([-np.array(raw, dtype=np.float32), np.array(raw, dtype=np.float32)])
        elif self.output_format == "scalar" and len(raw) == 1:
            return float(raw[0])
        return raw


# ===========================================================================
# Test Helpers
# ===========================================================================


def make_candidate(
    element_id: str,
    text: str = "",
    score: float = 0.5,
    element_type: ElementType = ElementType.TEXT,
    page_number: int = 1,
    source_filename: str = "paper.pdf",
    metadata: Optional[Dict[str, Any]] = None,
    related_element_ids: Optional[List[str]] = None,
    content: Optional[Any] = None,
) -> MultimodalSearchResult:
    """Creates a MultimodalSearchResult candidate with full provenance."""
    actual_content = content if content is not None else (text or f"Content for {element_id}")
    meta = dict(metadata or {})
    el = MultimodalElement(
        element_id=element_id,
        element_type=element_type,
        source_filename=source_filename,
        page_number=page_number,
        content=actual_content,
        metadata=meta,
        related_element_ids=list(related_element_ids or [f"rel_{element_id}"]),
    )
    return MultimodalSearchResult(
        element=el,
        score=score,
        element_id=el.element_id,
        element_type=el.element_type,
        source_filename=el.source_filename,
        page_number=el.page_number,
        related_element_ids=list(el.related_element_ids),
    )


# ===========================================================================
# 1. Base Interface and Contract Tests
# ===========================================================================


class TestBaseRerankerInterface:
    """Verifies BaseReranker interface definition and subclass contracts."""

    def test_cannot_instantiate_abstract_base_reranker(self):
        with pytest.raises(TypeError):
            BaseReranker()  # type: ignore

    def test_concrete_subclass_instantiation(self):
        class DummyReranker(BaseReranker):
            def rerank(self, query, candidates, top_k=None):
                return list(candidates)

        instance = DummyReranker()
        assert isinstance(instance, BaseReranker)
        c = make_candidate("c1", "text")
        res = instance.rerank("query", [c])
        assert res == [c]

    def test_callable_delegates_to_rerank(self):
        class EchoReranker(BaseReranker):
            def rerank(self, query, candidates, top_k=None):
                return list(candidates)[:top_k]

        reranker = EchoReranker()
        c1 = make_candidate("c1", "text1")
        c2 = make_candidate("c2", "text2")
        res = reranker("test query", [c1, c2], top_k=1)
        assert res == [c1]

    def test_cross_encoder_is_base_reranker_instance(self):
        fake_model = DeterministicFakeCrossEncoder()
        reranker = CrossEncoderReranker(model=fake_model)
        assert isinstance(reranker, BaseReranker)
        assert issubclass(CrossEncoderReranker, BaseReranker)

    def test_namespace_bridge_identity(self):
        assert RetBaseReranker is BaseReranker
        assert RetCrossEncoderReranker is CrossEncoderReranker


# ===========================================================================
# 2. Deterministic Reordering and Ranking Tests
# ===========================================================================


class TestCrossEncoderReordering:
    """Verifies that candidates are reordered strictly by cross-encoder score."""

    def test_candidates_reordered_by_score_descending(self):
        c1 = make_candidate("c1", "Attention is all you need", score=0.99)
        c2 = make_candidate("c2", "Convolutional neural networks", score=0.88)
        c3 = make_candidate("c3", "Transformer feed-forward layers", score=0.77)

        # Cross-encoder scores c3 highest, c1 second, c2 lowest
        fake_model = DeterministicFakeCrossEncoder(
            score_map={
                "Attention is all you need": 0.65,
                "Convolutional neural networks": 0.10,
                "Transformer feed-forward layers": 0.95,
            }
        )
        reranker = CrossEncoderReranker(model=fake_model)

        reranked = reranker.rerank("transformer attention", [c1, c2, c3])

        assert len(reranked) == 3
        assert [r.element_id for r in reranked] == ["c3", "c1", "c2"]
        assert [r.score for r in reranked] == [0.95, 0.65, 0.10]

    def test_reordering_from_reverse_initial_order(self):
        c_low = make_candidate("c_low", "low relevance", score=0.9)
        c_med = make_candidate("c_med", "medium relevance", score=0.8)
        c_high = make_candidate("c_high", "high relevance", score=0.1)

        fake_model = DeterministicFakeCrossEncoder(
            score_map={
                "low relevance": -1.5,
                "medium relevance": 0.0,
                "high relevance": 2.5,
            }
        )
        reranker = CrossEncoderReranker(model=fake_model)

        reranked = reranker.rerank("query", [c_low, c_med, c_high])

        assert [r.element_id for r in reranked] == ["c_high", "c_med", "c_low"]
        assert [r.score for r in reranked] == [2.5, 0.0, -1.5]

    def test_deterministic_tie_breaking_by_element_id(self):
        # All three candidates receive identical cross-encoder score 0.50
        c_z = make_candidate("doc_z", "same text", score=0.1)
        c_a = make_candidate("doc_a", "same text", score=0.2)
        c_m = make_candidate("doc_m", "same text", score=0.3)

        fake_model = DeterministicFakeCrossEncoder(default_score=0.50)
        reranker = CrossEncoderReranker(model=fake_model)

        reranked = reranker.rerank("query", [c_z, c_a, c_m])

        assert [r.element_id for r in reranked] == ["doc_a", "doc_m", "doc_z"]
        for r in reranked:
            assert r.score == 0.50


# ===========================================================================
# 3. Provenance and Attribute Preservation Tests
# ===========================================================================


class TestProvenancePreservation:
    """Verifies every original MultimodalSearchResult provenance field is strictly preserved."""

    def test_all_provenance_fields_preserved(self):
        orig_metadata = {
            "bbox": [10.0, 20.0, 100.0, 200.0],
            "section": "Methods",
            "custom_flag": True,
        }
        orig = make_candidate(
            element_id="provenance_elem_42",
            text="Detailed methodology text",
            score=0.123,
            element_type=ElementType.TEXT,
            page_number=7,
            source_filename="research_study.pdf",
            metadata=orig_metadata,
            related_element_ids=["rel_1", "rel_2"],
        )

        fake_model = DeterministicFakeCrossEncoder(
            score_map={"Detailed methodology text": 0.987}
        )
        reranker = CrossEncoderReranker(model=fake_model)

        results = reranker.rerank("methodology", [orig])
        assert len(results) == 1
        res = results[0]

        # 1. Element reference preserved exactly
        assert res.element is orig.element

        # 2. Score updated to cross-encoder score
        assert res.score == 0.987
        assert res.score != orig.score

        # 3. Provenance accessors strictly match
        assert res.element_id == "provenance_elem_42"
        assert res.element_type == ElementType.TEXT
        assert res.source_filename == "research_study.pdf"
        assert res.page_number == 7
        assert res.related_element_ids == ["rel_1", "rel_2"]

        # 4. Content and metadata properties strictly preserved
        assert res.content == "Detailed methodology text"
        assert res.metadata == orig_metadata
        assert res.metadata["custom_flag"] is True

    def test_original_input_list_not_mutated(self):
        c1 = make_candidate("c1", "text 1", score=0.1)
        c2 = make_candidate("c2", "text 2", score=0.2)
        candidates_input = [c1, c2]

        fake_model = DeterministicFakeCrossEncoder(default_score=0.88)
        reranker = CrossEncoderReranker(model=fake_model)

        reranker.rerank("query", candidates_input)

        # Original list order and objects unchanged
        assert candidates_input[0] is c1
        assert candidates_input[1] is c2
        assert c1.score == 0.1
        assert c2.score == 0.2


# ===========================================================================
# 4. top_k Parameter Slicing Tests
# ===========================================================================


class TestTopKFiltering:
    """Verifies top_k filtering behaviors."""

    def test_top_k_none_returns_all_candidates(self):
        candidates = [make_candidate(f"c{i}", f"text {i}") for i in range(5)]
        fake_model = DeterministicFakeCrossEncoder(default_score=0.5)
        reranker = CrossEncoderReranker(model=fake_model)

        res = reranker.rerank("query", candidates, top_k=None)
        assert len(res) == 5

    def test_top_k_slices_highest_ranked_results(self):
        c1 = make_candidate("c1", "low", score=0.1)
        c2 = make_candidate("c2", "high", score=0.2)
        c3 = make_candidate("c3", "mid", score=0.3)

        fake_model = DeterministicFakeCrossEncoder(
            score_map={"high": 0.9, "mid": 0.5, "low": 0.1}
        )
        reranker = CrossEncoderReranker(model=fake_model)

        res = reranker.rerank("query", [c1, c2, c3], top_k=2)

        assert len(res) == 2
        assert [r.element_id for r in res] == ["c2", "c3"]
        assert [r.score for r in res] == [0.9, 0.5]

    def test_top_k_greater_than_candidate_count(self):
        candidates = [make_candidate("c1", "text 1"), make_candidate("c2", "text 2")]
        fake_model = DeterministicFakeCrossEncoder(default_score=0.5)
        reranker = CrossEncoderReranker(model=fake_model)

        res = reranker.rerank("query", candidates, top_k=100)
        assert len(res) == 2

    def test_top_k_one_returns_single_best(self):
        c1 = make_candidate("c1", "text 1")
        c2 = make_candidate("c2", "text 2")
        fake_model = DeterministicFakeCrossEncoder(
            score_map={"text 1": 0.1, "text 2": 0.9}
        )
        reranker = CrossEncoderReranker(model=fake_model)

        res = reranker.rerank("query", [c1, c2], top_k=1)
        assert len(res) == 1
        assert res[0].element_id == "c2"


# ===========================================================================
# 5. Input Validation Tests
# ===========================================================================


class TestInputValidations:
    """Verifies error handling for invalid queries, top_k, candidates, and batch_size."""

    @pytest.fixture
    def reranker(self) -> CrossEncoderReranker:
        return CrossEncoderReranker(model=DeterministicFakeCrossEncoder())

    def test_query_none_raises_type_error(self, reranker: CrossEncoderReranker):
        candidates = [make_candidate("c1", "text")]
        with pytest.raises(TypeError, match="query must be a string"):
            reranker.rerank(None, candidates)  # type: ignore

    def test_query_non_string_raises_type_error(self, reranker: CrossEncoderReranker):
        candidates = [make_candidate("c1", "text")]
        with pytest.raises(TypeError, match="query must be a string"):
            reranker.rerank(12345, candidates)  # type: ignore

        with pytest.raises(TypeError, match="query must be a string"):
            reranker.rerank(["invalid", "query"], candidates)  # type: ignore

    def test_query_empty_string_raises_value_error(self, reranker: CrossEncoderReranker):
        candidates = [make_candidate("c1", "text")]
        with pytest.raises(ValueError, match="query cannot be empty"):
            reranker.rerank("", candidates)

    def test_query_whitespace_only_raises_value_error(self, reranker: CrossEncoderReranker):
        candidates = [make_candidate("c1", "text")]
        with pytest.raises(ValueError, match="query cannot be empty"):
            reranker.rerank("   \t\n  ", candidates)

    def test_candidates_none_raises_type_error(self, reranker: CrossEncoderReranker):
        with pytest.raises(TypeError, match="candidates cannot be None"):
            reranker.rerank("valid query", None)  # type: ignore

    def test_candidates_non_sequence_raises_type_error(self, reranker: CrossEncoderReranker):
        with pytest.raises(TypeError, match="candidates must be a sequence"):
            reranker.rerank("valid query", "not_a_sequence")  # type: ignore

    def test_candidate_item_invalid_type_raises_type_error(self, reranker: CrossEncoderReranker):
        c1 = make_candidate("c1", "text")
        with pytest.raises(TypeError, match="must be a MultimodalSearchResult instance"):
            reranker.rerank("valid query", [c1, "not_a_candidate"])  # type: ignore

    def test_top_k_non_positive_raises_value_error(self, reranker: CrossEncoderReranker):
        candidates = [make_candidate("c1", "text")]
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            reranker.rerank("query", candidates, top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            reranker.rerank("query", candidates, top_k=-3)

    def test_top_k_invalid_type_raises_type_error(self, reranker: CrossEncoderReranker):
        candidates = [make_candidate("c1", "text")]
        with pytest.raises(TypeError, match="top_k must be an integer"):
            reranker.rerank("query", candidates, top_k=2.5)  # type: ignore

        with pytest.raises(TypeError, match="top_k must be an integer"):
            reranker.rerank("query", candidates, top_k="five")  # type: ignore

        with pytest.raises(TypeError, match="top_k must be an integer"):
            reranker.rerank("query", candidates, top_k=True)  # type: ignore

    def test_init_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            CrossEncoderReranker(model=DeterministicFakeCrossEncoder(), batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            CrossEncoderReranker(model=DeterministicFakeCrossEncoder(), batch_size=-5)

        with pytest.raises(TypeError, match="batch_size must be an integer"):
            CrossEncoderReranker(model=DeterministicFakeCrossEncoder(), batch_size="32")  # type: ignore

        with pytest.raises(TypeError, match="batch_size must be an integer"):
            CrossEncoderReranker(model=DeterministicFakeCrossEncoder(), batch_size=False)  # type: ignore

    def test_init_invalid_model_name(self):
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            CrossEncoderReranker(model_name="")

        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            CrossEncoderReranker(model_name="   ")

    def test_init_invalid_max_length(self):
        with pytest.raises(ValueError, match="max_length must be positive"):
            CrossEncoderReranker(max_length=0)

        with pytest.raises(TypeError, match="max_length must be an integer"):
            CrossEncoderReranker(max_length=1.5)  # type: ignore


# ===========================================================================
# 6. Edge Cases & Score-Count Mismatch Handling
# ===========================================================================


class TestEdgeCasesAndMismatch:
    """Verifies edge cases like empty candidates and score-count mismatches."""

    def test_empty_candidates_returns_empty_list_without_model_invocation(self):
        fake_model = DeterministicFakeCrossEncoder()
        reranker = CrossEncoderReranker(model=fake_model)

        res = reranker.rerank("attention", [])

        assert res == []
        assert len(fake_model.call_history) == 0

    def test_single_candidate_reranking(self):
        c = make_candidate("single", "single text", score=0.2)
        fake_model = DeterministicFakeCrossEncoder(score_map={"single text": 0.88})
        reranker = CrossEncoderReranker(model=fake_model)

        res = reranker.rerank("query", [c])

        assert len(res) == 1
        assert res[0].element_id == "single"
        assert res[0].score == 0.88

    def test_model_score_count_mismatch_fewer_scores(self):
        c1 = make_candidate("c1", "text 1")
        c2 = make_candidate("c2", "text 2")
        c3 = make_candidate("c3", "text 3")

        # Fake returns only 2 scores for 3 candidates
        fake_model = DeterministicFakeCrossEncoder(fixed_scores=[0.9, 0.5])
        reranker = CrossEncoderReranker(model=fake_model)

        with pytest.raises(ScoreCountMismatchError, match="returned 2 scores for 3 candidates"):
            reranker.rerank("query", [c1, c2, c3])

        # Verify it also satisfies general RuntimeError and ValueError checks
        with pytest.raises(RuntimeError):
            reranker.rerank("query", [c1, c2, c3])
        with pytest.raises(ValueError):
            reranker.rerank("query", [c1, c2, c3])

    def test_model_score_count_mismatch_more_scores(self):
        c1 = make_candidate("c1", "text 1")
        # Fake returns 3 scores for 1 candidate
        fake_model = DeterministicFakeCrossEncoder(fixed_scores=[0.9, 0.5, 0.1])
        reranker = CrossEncoderReranker(model=fake_model)

        with pytest.raises(ScoreCountMismatchError, match="returned 3 scores for 1 candidates"):
            reranker.rerank("query", [c1])

    def test_model_returns_none_raises_mismatch(self):
        class NoneReturningModel:
            def predict(self, pairs, **kwargs):
                return None

        reranker = CrossEncoderReranker(model=NoneReturningModel())
        c = make_candidate("c1", "text")

        with pytest.raises(ScoreCountMismatchError, match="returned None"):
            reranker.rerank("query", [c])


# ===========================================================================
# 7. Model Output Format Normalization Tests
# ===========================================================================


class TestScoreNormalization:
    """Verifies that various model output formats are normalized correctly."""

    def test_numpy_1d_array_scores(self):
        c1 = make_candidate("c1", "t1")
        c2 = make_candidate("c2", "t2")
        fake_model = DeterministicFakeCrossEncoder(
            fixed_scores=[0.3, 0.7], output_format="numpy_1d"
        )
        reranker = CrossEncoderReranker(model=fake_model)

        res = reranker.rerank("q", [c1, c2])
        assert [r.element_id for r in res] == ["c2", "c1"]
        assert [r.score for r in res] == pytest.approx([0.7, 0.3])

    def test_numpy_2d_column_array_scores(self):
        c1 = make_candidate("c1", "t1")
        c2 = make_candidate("c2", "t2")
        fake_model = DeterministicFakeCrossEncoder(
            fixed_scores=[0.1, 0.9], output_format="numpy_2d_col"
        )
        reranker = CrossEncoderReranker(model=fake_model)

        res = reranker.rerank("q", [c1, c2])
        assert [r.element_id for r in res] == ["c2", "c1"]
        assert [r.score for r in res] == pytest.approx([0.9, 0.1])

    def test_numpy_2d_logits_array_scores(self):
        c1 = make_candidate("c1", "t1")
        c2 = make_candidate("c2", "t2")
        fake_model = DeterministicFakeCrossEncoder(
            fixed_scores=[0.25, 0.85], output_format="numpy_2d_logits"
        )
        reranker = CrossEncoderReranker(model=fake_model)

        res = reranker.rerank("q", [c1, c2])
        assert [r.element_id for r in res] == ["c2", "c1"]
        assert [r.score for r in res] == pytest.approx([0.85, 0.25])

    def test_scalar_score_single_candidate(self):
        c = make_candidate("c1", "t1")
        fake_model = DeterministicFakeCrossEncoder(
            fixed_scores=[0.42], output_format="scalar"
        )
        reranker = CrossEncoderReranker(model=fake_model)

        res = reranker.rerank("q", [c])
        assert len(res) == 1
        assert res[0].score == 0.42


# ===========================================================================
# 8. Multimodal Elements & Text Extraction Tests
# ===========================================================================


class TestMultimodalCandidateHandling:
    """Verifies that heterogeneous multimodal candidate types are safely extracted."""

    def test_heterogeneous_modalities_reranked(self):
        c_text = make_candidate("elem_text", "Transformer text block", element_type=ElementType.TEXT)
        c_caption = make_candidate("elem_caption", "Figure 1: Architecture diagram", element_type=ElementType.CAPTION)
        c_table = make_candidate("elem_table", "| Layer | Dim |\n| Attn | 512 |", element_type=ElementType.TABLE)
        c_eq = make_candidate("elem_eq", "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V", element_type=ElementType.EQUATION)

        # Image element with caption in metadata and binary content
        c_img = make_candidate(
            "elem_img",
            content=b"binary_image_data_bytes",
            element_type=ElementType.IMAGE,
            metadata={"caption": "Heatmap of self-attention weights"},
        )

        fake_model = DeterministicFakeCrossEncoder(
            score_map={
                "Transformer text block": 0.4,
                "Figure 1: Architecture diagram": 0.3,
                "| Layer | Dim |\n| Attn | 512 |": 0.2,
                "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V": 0.5,
                "Heatmap of self-attention weights": 0.9,
            }
        )
        reranker = CrossEncoderReranker(model=fake_model)

        res = reranker.rerank("attention heatmap", [c_text, c_caption, c_table, c_eq, c_img])

        assert len(res) == 5
        # Top result is the image resolved through its metadata caption
        assert res[0].element_id == "elem_img"
        assert res[0].score == 0.9
        assert res[0].element_type == ElementType.IMAGE
        # Content is still original binary bytes, not corrupted
        assert res[0].content == b"binary_image_data_bytes"

        # Equation is second
        assert res[1].element_id == "elem_eq"
        assert res[1].score == 0.5

    def test_image_without_metadata_text_does_not_crash(self):
        c_img_bare = make_candidate(
            "elem_img_bare",
            content=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
            element_type=ElementType.IMAGE,
            metadata={},
        )
        fake_model = DeterministicFakeCrossEncoder(score_map={"": 0.05})
        reranker = CrossEncoderReranker(model=fake_model)

        res = reranker.rerank("query", [c_img_bare])
        assert len(res) == 1
        assert res[0].element_id == "elem_img_bare"
        assert res[0].score == 0.05
        # Verified pairs sent to model
        pairs_sent = fake_model.call_history[0]
        assert pairs_sent == [("query", "")]

    def test_custom_text_extractor_injection(self):
        c1 = make_candidate("c1", "ignored text", metadata={"custom_title": "Primary Title"})
        c2 = make_candidate("c2", "ignored text", metadata={"custom_title": "Secondary Title"})

        def custom_extractor(candidate: MultimodalSearchResult) -> str:
            return str(candidate.metadata.get("custom_title", ""))

        fake_model = DeterministicFakeCrossEncoder(
            score_map={"Primary Title": 0.95, "Secondary Title": 0.35}
        )
        reranker = CrossEncoderReranker(
            model=fake_model,
            text_extractor=custom_extractor,
        )

        res = reranker.rerank("query", [c1, c2])
        assert [r.element_id for r in res] == ["c1", "c2"]
        assert [r.score for r in res] == [0.95, 0.35]
        assert fake_model.call_history[0] == [("query", "Primary Title"), ("query", "Secondary Title")]


# ===========================================================================
# 9. Convenience Function Tests
# ===========================================================================


class TestConvenienceFunction:
    """Verifies rerank_cross_encoder standalone convenience function."""

    def test_rerank_cross_encoder_with_injected_model(self):
        c1 = make_candidate("c1", "first")
        c2 = make_candidate("c2", "second")

        fake_model = DeterministicFakeCrossEncoder(
            score_map={"first": 0.2, "second": 0.8}
        )

        res = rerank_cross_encoder(
            query="query",
            candidates=[c1, c2],
            model=fake_model,
            top_k=1,
        )

        assert len(res) == 1
        assert res[0].element_id == "c2"
        assert res[0].score == 0.8

    def test_rerank_cross_encoder_with_existing_reranker(self):
        c1 = make_candidate("c1", "first")
        c2 = make_candidate("c2", "second")

        fake_model = DeterministicFakeCrossEncoder(
            score_map={"first": 0.9, "second": 0.1}
        )
        reranker = CrossEncoderReranker(model=fake_model)

        res = rerank_cross_encoder(
            query="query",
            candidates=[c1, c2],
            reranker=reranker,
        )

        assert len(res) == 2
        assert [r.element_id for r in res] == ["c1", "c2"]
