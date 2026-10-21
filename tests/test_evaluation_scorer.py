"""Targeted unit tests for the deterministic retrieval scoring module."""

from pathlib import Path
import pytest

from app.evaluation import (
    ConfigurationScore,
    EvaluationQuestion,
    GoldEvidence,
    QuestionScore,
    QuestionRetrievalRun,
    RetrievalBenchmarkScoreReport,
    get_retrieval_evaluation_dataset,
    score_baseline_json,
    score_evaluation_runs,
    score_question,
    score_retrieval_results,
)


def test_score_retrieval_results_single_gold():
    """Verify single gold element ranking and partial/complete recall."""
    gold_ids = ["chunk_A"]

    # Rank 1 hit
    res1 = ["chunk_A", "chunk_B", "chunk_C"]
    s1 = score_retrieval_results(res1, gold_ids, configuration="test")
    assert s1.partial_evidence_recall == 1
    assert s1.complete_evidence_recall == 1
    assert s1.gold_ranks == {"chunk_A": 1}
    assert s1.min_gold_rank == 1
    assert s1.reciprocal_rank == 1.0

    # Rank 3 hit
    res2 = ["chunk_X", "chunk_Y", "chunk_A"]
    s2 = score_retrieval_results(res2, gold_ids, configuration="test")
    assert s2.partial_evidence_recall == 1
    assert s2.complete_evidence_recall == 1
    assert s2.gold_ranks == {"chunk_A": 3}
    assert s2.min_gold_rank == 3
    assert s2.reciprocal_rank == pytest.approx(1.0 / 3.0)

    # Miss
    res3 = ["chunk_X", "chunk_Y", "chunk_Z"]
    s3 = score_retrieval_results(res3, gold_ids, configuration="test")
    assert s3.partial_evidence_recall == 0
    assert s3.complete_evidence_recall == 0
    assert s3.gold_ranks == {"chunk_A": None}
    assert s3.min_gold_rank is None
    assert s3.reciprocal_rank == 0.0


def test_score_retrieval_results_multi_gold_partial_and_complete():
    """Verify multi-gold distinction where partial=1 and complete=0 on partial recall."""
    gold_ids = ["chunk_A", "chunk_B"]

    # Partial recall (only chunk_A retrieved)
    partial_res = ["chunk_A", "chunk_X", "chunk_Y"]
    s_part = score_retrieval_results(partial_res, gold_ids, configuration="test")
    assert s_part.partial_evidence_recall == 1
    assert s_part.complete_evidence_recall == 0
    assert s_part.gold_ranks == {"chunk_A": 1, "chunk_B": None}
    assert s_part.min_gold_rank == 1
    assert s_part.retrieved_gold_elements == 1
    assert s_part.total_gold_elements == 2

    # Complete recall (both chunks retrieved)
    complete_res = ["chunk_X", "chunk_B", "chunk_A"]
    s_comp = score_retrieval_results(complete_res, gold_ids, configuration="test")
    assert s_comp.partial_evidence_recall == 1
    assert s_comp.complete_evidence_recall == 1
    assert s_comp.gold_ranks == {"chunk_A": 3, "chunk_B": 2}
    assert s_comp.min_gold_rank == 2
    assert s_comp.retrieved_gold_elements == 2
    assert s_comp.total_gold_elements == 2

    # Zero recall
    zero_res = ["chunk_X", "chunk_Y", "chunk_Z"]
    s_zero = score_retrieval_results(zero_res, gold_ids, configuration="test")
    assert s_zero.partial_evidence_recall == 0
    assert s_zero.complete_evidence_recall == 0
    assert s_zero.gold_ranks == {"chunk_A": None, "chunk_B": None}


def test_q3_distractor_excluded_from_positive_gold():
    """Verify Q3's reference [22] distractor is excluded from positive scoring."""
    dataset = get_retrieval_evaluation_dataset()
    q3 = dataset["Q3"]

    # If only distractor chunk is retrieved, score must be 0
    distractor_only = ["sample_paper.pdf#chunk_41", "other_chunk"]
    q_score1 = score_question(q3, {"dense": distractor_only})
    assert q_score1.scores["dense"].partial_evidence_recall == 0
    assert q_score1.scores["dense"].complete_evidence_recall == 0
    assert "sample_paper.pdf#chunk_41" not in q_score1.scores["dense"].gold_ranks

    # If primary positive chunk is retrieved, score is 1
    primary_hit = ["sample_paper.pdf#chunk_23", "sample_paper.pdf#chunk_41"]
    q_score2 = score_question(q3, {"dense": primary_hit})
    assert q_score2.scores["dense"].partial_evidence_recall == 1
    assert q_score2.scores["dense"].complete_evidence_recall == 1
    assert q_score2.scores["dense"].gold_ranks == {"sample_paper.pdf#chunk_23": 1}


def test_q6_caption_element_is_positive_gold():
    """Verify Q6 caption element_id is treated as a positive gold element."""
    dataset = get_retrieval_evaluation_dataset()
    q6 = dataset["Q6"]

    res = ["sample_paper.pdf#page_2_cap_1", "other_chunk"]
    q_score = score_question(q6, {"hybrid_rrf": res})
    assert q_score.scores["hybrid_rrf"].partial_evidence_recall == 1
    assert q_score.scores["hybrid_rrf"].complete_evidence_recall == 1
    assert q_score.scores["hybrid_rrf"].gold_ranks == {"sample_paper.pdf#page_2_cap_1": 1}


def test_q1_and_q4_multi_gold_scoring():
    """Verify Q1 and Q4 retain is_multi_gold=True and multiple gold elements."""
    dataset = get_retrieval_evaluation_dataset()
    q1 = dataset["Q1"]
    q4 = dataset["Q4"]

    assert len(q1.unique_gold_element_ids) == 2
    assert len(q4.unique_gold_element_ids) == 2

    # Q1 with partial retrieval
    res_q1_partial = ["sample_paper.pdf#chunk_10", "other_chunk"]
    score_q1 = score_question(q1, {"bm25": res_q1_partial})
    assert score_q1.is_multi_gold is True
    assert score_q1.scores["bm25"].partial_evidence_recall == 1
    assert score_q1.scores["bm25"].complete_evidence_recall == 0

    # Q4 with complete retrieval
    res_q4_complete = ["sample_paper.pdf#chunk_21", "sample_paper.pdf#chunk_22"]
    score_q4 = score_question(q4, {"dense": res_q4_complete})
    assert score_q4.is_multi_gold is True
    assert score_q4.scores["dense"].partial_evidence_recall == 1
    assert score_q4.scores["dense"].complete_evidence_recall == 1


def test_score_baseline_json_against_real_results():
    """Verify score_baseline_json parses data/evaluation/baseline_retrieval_results.json accurately."""
    json_path = Path("data/evaluation/baseline_retrieval_results.json")
    assert json_path.is_file(), "baseline_retrieval_results.json must exist"

    report = score_baseline_json(json_path)
    assert report.total_questions == 8
    assert set(report.configurations) == {"dense", "bm25", "hybrid_rrf", "hybrid_reranked"}

    # Dense FAISS
    dense_agg = report.aggregate_scores["dense"]
    assert dense_agg.partial_recall_count == 8
    assert dense_agg.complete_recall_count == 8
    assert dense_agg.complete_recall_rate == 1.0

    # BM25: missed chunk_10 on Q1, so complete recall is 7/8
    bm25_agg = report.aggregate_scores["bm25"]
    assert bm25_agg.partial_recall_count == 8
    assert bm25_agg.complete_recall_count == 7
    assert bm25_agg.complete_recall_rate == 0.875

    # Hybrid RRF
    rrf_agg = report.aggregate_scores["hybrid_rrf"]
    assert rrf_agg.partial_recall_count == 8
    assert rrf_agg.complete_recall_count == 8
    assert rrf_agg.complete_recall_rate == 1.0

    # Hybrid RRF + Cross-Encoder: highest MRR
    rerank_agg = report.aggregate_scores["hybrid_reranked"]
    assert rerank_agg.partial_recall_count == 8
    assert rerank_agg.complete_recall_count == 8
    assert rerank_agg.mrr > rrf_agg.mrr
    assert rerank_agg.mrr > dense_agg.mrr
    assert rerank_agg.mrr > bm25_agg.mrr
    assert rerank_agg.mrr == pytest.approx(0.9375)

    # Format report check
    table_str = report.format_summary_table()
    assert "RETRIEVAL EVALUATION BENCHMARK — AGGREGATE SUMMARY" in table_str
    assert "hybrid_reranked" in table_str
