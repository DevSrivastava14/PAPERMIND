"""Focused unit tests for Day 15 Task 3: Reciprocal Rank Fusion (RRF).

Verifies:
- Correct rank contribution formula (1 / (k + rank))
- Overlapping results receive combined rank contributions
- Deduplication by element_id
- Dense-only results survive
- Sparse-only results survive
- Provenance preservation (element, metadata, relationships, page, content)
- Original FAISS/BM25 scores are replaced with pure RRF scores
- Deterministic ordering with tie-breaking
- Empty inputs handling
- Input and parameter validation (invalid k, invalid top_k, malformed lists)
- Synthetic independence from actual FAISS/BM25 execution
"""

import pytest

from app.indexing.multimodal_models import MultimodalSearchResult
from app.multimodal.representation import ElementType, MultimodalElement
from app.retrieval.rrf import (
    DEFAULT_RRF_K,
    ReciprocalRankFusion,
    reciprocal_rank_fusion,
)


# ===========================================================================
# Test Helpers
# ===========================================================================


def make_result(
    element_id: str,
    score: float = 1.0,
    element_type: ElementType = ElementType.TEXT,
    page_number: int = 1,
) -> MultimodalSearchResult:
    """Creates a synthetic MultimodalSearchResult for isolated RRF unit testing."""
    element = MultimodalElement(
        element_id=element_id,
        element_type=element_type,
        source_filename="synthetic_paper.pdf",
        page_number=page_number,
        content=f"Content for {element_id}",
        metadata={"orig_score": score, "test_key": f"val_{element_id}"},
        related_element_ids=[f"rel_{element_id}"],
    )
    return MultimodalSearchResult(
        element=element,
        score=score,
        element_id=element.element_id,
        element_type=element.element_type,
        source_filename=element.source_filename,
        page_number=element.page_number,
        related_element_ids=list(element.related_element_ids),
    )


# ===========================================================================
# 1. Formula & Rank Contribution Tests
# ===========================================================================


class TestRRFRankContribution:
    """Verifies that RRF score formula: sum(1 / (k + rank)) is applied accurately."""

    def test_single_list_rank_contributions_default_k(self):
        r1 = make_result("doc-1", score=0.95)
        r2 = make_result("doc-2", score=0.85)
        r3 = make_result("doc-3", score=0.75)

        # Default k=60
        fused = reciprocal_rank_fusion([[r1, r2, r3]])

        assert len(fused) == 3
        # 1-based ranks: rank 1 -> 1/(60+1), rank 2 -> 1/(60+2), rank 3 -> 1/(60+3)
        assert pytest.approx(fused[0].score, rel=1e-6) == 1.0 / 61.0
        assert pytest.approx(fused[1].score, rel=1e-6) == 1.0 / 62.0
        assert pytest.approx(fused[2].score, rel=1e-6) == 1.0 / 63.0

    def test_single_list_rank_contributions_custom_k(self):
        r1 = make_result("doc-1")
        r2 = make_result("doc-2")

        # Custom k=10
        fused = reciprocal_rank_fusion([[r1, r2]], k=10)

        assert len(fused) == 2
        assert pytest.approx(fused[0].score, rel=1e-6) == 1.0 / 11.0
        assert pytest.approx(fused[1].score, rel=1e-6) == 1.0 / 12.0

    def test_overlapping_results_receive_combined_contribution(self):
        """Elements appearing in multiple ranked lists accumulate contributions."""
        # Dense list: A at rank 1, B at rank 2
        dense = [make_result("elem-A", score=0.9), make_result("elem-B", score=0.8)]
        # Sparse list: B at rank 1, C at rank 2
        sparse = [make_result("elem-B", score=12.5), make_result("elem-C", score=8.1)]

        k = 60
        fused = reciprocal_rank_fusion([dense, sparse], k=k)

        # Expected scores:
        # elem-B: 1/(60+2) [from dense rank 2] + 1/(60+1) [from sparse rank 1]
        expected_score_b = (1.0 / 62.0) + (1.0 / 61.0)
        # elem-A: 1/(60+1) [from dense rank 1]
        expected_score_a = 1.0 / 61.0
        # elem-C: 1/(60+2) [from sparse rank 2]
        expected_score_c = 1.0 / 62.0

        assert len(fused) == 3
        assert fused[0].element_id == "elem-B"
        assert fused[1].element_id == "elem-A"
        assert fused[2].element_id == "elem-C"

        assert pytest.approx(fused[0].score, rel=1e-6) == expected_score_b
        assert pytest.approx(fused[1].score, rel=1e-6) == expected_score_a
        assert pytest.approx(fused[2].score, rel=1e-6) == expected_score_c


# ===========================================================================
# 2. Deduplication and Single-List Survival Tests
# ===========================================================================


class TestDeduplicationAndSurvival:
    """Verifies deduplication by element_id and survival of dense-only / sparse-only items."""

    def test_deduplication_by_element_id_across_lists(self):
        dense = [make_result("overlap-id", score=0.99)]
        sparse = [make_result("overlap-id", score=15.0)]

        fused = reciprocal_rank_fusion([dense, sparse])
        assert len(fused) == 1
        assert fused[0].element_id == "overlap-id"

    def test_deduplication_within_same_list_uses_best_rank(self):
        # A list erroneously containing duplicate element_id
        duplicate_list = [
            make_result("elem-dup", score=0.9),
            make_result("elem-dup", score=0.5),
        ]
        fused = reciprocal_rank_fusion([duplicate_list], k=60)
        assert len(fused) == 1
        # Only rank 1 contribution: 1/61
        assert pytest.approx(fused[0].score, rel=1e-6) == 1.0 / 61.0

    def test_dense_only_results_survive(self):
        dense = [make_result("dense-only-1"), make_result("dense-only-2")]
        sparse = [make_result("other-item")]

        fused = reciprocal_rank_fusion([dense, sparse], k=60)
        element_ids = [res.element_id for res in fused]

        assert "dense-only-1" in element_ids
        assert "dense-only-2" in element_ids

    def test_sparse_only_results_survive(self):
        dense = [make_result("other-item")]
        sparse = [make_result("sparse-only-1"), make_result("sparse-only-2")]

        fused = reciprocal_rank_fusion([dense, sparse], k=60)
        element_ids = [res.element_id for res in fused]

        assert "sparse-only-1" in element_ids
        assert "sparse-only-2" in element_ids


# ===========================================================================
# 3. Provenance and Score Replacement Tests
# ===========================================================================


class TestProvenanceAndScoreReplacement:
    """Verifies that original provenance is preserved and original scores are replaced."""

    def test_original_faiss_bm25_scores_replaced(self):
        res_dense = make_result("doc-1", score=0.888)
        res_sparse = make_result("doc-1", score=18.45)

        fused = reciprocal_rank_fusion([[res_dense], [res_sparse]], k=60)
        assert len(fused) == 1
        fused_res = fused[0]

        # Score must NOT be 0.888 or 18.45
        assert fused_res.score != 0.888
        assert fused_res.score != 18.45
        assert pytest.approx(fused_res.score, rel=1e-6) == (1.0 / 61.0) * 2.0

    def test_full_provenance_preservation(self):
        orig_el = MultimodalElement(
            element_id="table-element-77",
            element_type=ElementType.TABLE,
            source_filename="deep_learning_benchmark.pdf",
            page_number=5,
            content="| Model | Acc |\n| ResNet | 76.1 |",
            metadata={"source": "benchmark_table", "rows": 2},
            related_element_ids=["cap-77"],
        )
        orig_res = MultimodalSearchResult(
            element=orig_el,
            score=0.912,
            element_id=orig_el.element_id,
            element_type=orig_el.element_type,
            source_filename=orig_el.source_filename,
            page_number=orig_el.page_number,
            related_element_ids=list(orig_el.related_element_ids),
        )

        fused = reciprocal_rank_fusion([[orig_res]], k=60)
        assert len(fused) == 1
        res = fused[0]

        # Exact MultimodalElement preservation
        assert res.element is orig_el
        assert res.element_id == "table-element-77"
        assert res.element_type == ElementType.TABLE
        assert res.source_filename == "deep_learning_benchmark.pdf"
        assert res.page_number == 5
        assert res.metadata == {"source": "benchmark_table", "rows": 2}
        assert res.related_element_ids == ["cap-77"]
        assert res.content == "| Model | Acc |\n| ResNet | 76.1 |"


# ===========================================================================
# 4. Deterministic Ordering and Tie-Breaking Tests
# ===========================================================================


class TestDeterministicOrdering:
    """Verifies that results are sorted descending by RRF score, with deterministic tie-breaking."""

    def test_descending_rrf_ordering(self):
        list_1 = [make_result(f"id-{i}") for i in range(5)]
        fused = reciprocal_rank_fusion([list_1], k=60)

        for i in range(len(fused) - 1):
            assert fused[i].score > fused[i + 1].score

    def test_deterministic_tie_breaking(self):
        """When two elements have identical RRF scores, tie-break deterministically by element_id."""
        # List 1 has 'elem-Z' at rank 1
        # List 2 has 'elem-A' at rank 1
        # Neither appears in the other list.
        # Both get identical RRF score: 1/(60+1).
        list_1 = [make_result("elem-Z")]
        list_2 = [make_result("elem-A")]

        fused = reciprocal_rank_fusion([list_1, list_2], k=60)

        assert len(fused) == 2
        assert fused[0].score == fused[1].score
        # Tie-break alphabetically by element_id ascending: 'elem-A' before 'elem-Z'
        assert fused[0].element_id == "elem-A"
        assert fused[1].element_id == "elem-Z"


# ===========================================================================
# 5. Empty Inputs & Edge Cases
# ===========================================================================


class TestEmptyInputsAndEdgeCases:
    """Verifies that empty result lists and combinations are handled gracefully."""

    def test_empty_ranked_lists_returns_empty_list(self):
        assert reciprocal_rank_fusion([]) == []

    def test_all_empty_sublists_returns_empty_list(self):
        assert reciprocal_rank_fusion([[], [], []]) == []

    def test_one_empty_list_and_one_populated_list(self):
        populated = [make_result("el-1"), make_result("el-2")]
        fused = reciprocal_rank_fusion([[], populated], k=60)

        assert len(fused) == 2
        assert fused[0].element_id == "el-1"
        assert fused[1].element_id == "el-2"

    def test_top_k_limiting(self):
        r_list = [make_result(f"id-{i}") for i in range(10)]
        fused = reciprocal_rank_fusion([r_list], k=60, top_k=3)
        assert len(fused) == 3

    def test_fewer_results_than_top_k(self):
        r_list = [make_result("id-1"), make_result("id-2")]
        fused = reciprocal_rank_fusion([r_list], k=60, top_k=10)
        assert len(fused) == 2


# ===========================================================================
# 6. Parameter Validation Tests
# ===========================================================================


class TestRRFValidation:
    """Verifies that invalid k, top_k, and malformed inputs are cleanly rejected."""

    def test_default_k_value(self):
        fuser = ReciprocalRankFusion()
        assert fuser.k == DEFAULT_RRF_K
        assert fuser.k == 60

    def test_invalid_k_values_raise_value_error(self):
        with pytest.raises(ValueError, match="k must be a positive number"):
            ReciprocalRankFusion(k=0)

        with pytest.raises(ValueError, match="k must be a positive number"):
            ReciprocalRankFusion(k=-10)

    def test_invalid_k_types_raise_type_error(self):
        with pytest.raises(TypeError, match="k must be an integer or float"):
            ReciprocalRankFusion(k="60")

        with pytest.raises(TypeError, match="k must be an integer or float"):
            ReciprocalRankFusion(k=True)

        with pytest.raises(TypeError, match="k must be an integer or float"):
            ReciprocalRankFusion(k=None)

    def test_invalid_top_k_raises_error(self):
        fuser = ReciprocalRankFusion()
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            fuser.fuse([[make_result("el-1")]], top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            fuser.fuse([[make_result("el-1")]], top_k=-1)

        with pytest.raises(TypeError, match="top_k must be an integer"):
            fuser.fuse([[make_result("el-1")]], top_k="5")

    def test_none_ranked_lists_raises_type_error(self):
        fuser = ReciprocalRankFusion()
        with pytest.raises(TypeError, match="ranked_lists cannot be None"):
            fuser.fuse(None)

    def test_non_sequence_ranked_lists_raises_type_error(self):
        fuser = ReciprocalRankFusion()
        with pytest.raises(TypeError, match="ranked_lists must be a sequence"):
            fuser.fuse(123)

    def test_none_or_invalid_item_in_ranked_lists(self):
        fuser = ReciprocalRankFusion()
        with pytest.raises(TypeError, match="cannot be None"):
            fuser.fuse([None])

        with pytest.raises(TypeError, match="must be a sequence"):
            fuser.fuse(["not_a_sequence"])

        with pytest.raises(TypeError, match="must be a MultimodalSearchResult"):
            fuser.fuse([["string_instead_of_result"]])
