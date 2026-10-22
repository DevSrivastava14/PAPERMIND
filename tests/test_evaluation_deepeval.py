"""Targeted unit tests for DeepEval semantic retrieval evaluation layer.

Tests use mock metrics and offline stubbing so that no remote LLM API calls
or credentials are required.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from app.evaluation import (
    DeepEvalAnswerEvaluator,
    DeepEvalBenchmarkReport,
    DeepEvalConfigurationResult,
    DeepEvalMetricResult,
    DeepEvalQuestionResult,
    DeepEvalRetrievalEvaluator,
    EvaluationQuestion,
    GoldEvidence,
    QuestionRetrievalRun,
    RetrievalConfiguration,
    build_llm_test_case,
    extract_retrieved_contexts,
    get_retrieval_evaluation_dataset,
    normalize_metric_name,
)
from app.evaluation.deepeval_evaluator import is_groq_rate_limit_error
from app.indexing.multimodal_models import MultimodalSearchResult
from app.multimodal.representation import ElementType, MultimodalElement


class MockRetrievalMetric(BaseMetric):
    """Stubbed DeepEval metric for isolated offline testing."""

    def __init__(
        self,
        name: str = "contextual_relevancy",
        score: float = 0.85,
        threshold: float = 0.5,
        reason: str = "Retrieved contexts are relevant to query",
        should_fail: bool = False,
    ) -> None:
        super().__init__()
        self._name = name
        self.score = score
        self.threshold = threshold
        self.reason = reason
        self.should_fail = should_fail
        self.called_with: list[LLMTestCase] = []

    @property
    def __name__(self) -> str:
        return self._name

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        if self.should_fail:
            raise RuntimeError("Simulated metric execution failure")
        self.called_with.append(test_case)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)


class MockAnswerMetric(BaseMetric):
    """Answer metric with a deterministic sequence of local outcomes."""

    def __init__(self, name: str, outcomes: list[object]) -> None:
        super().__init__()
        self._name = name
        self.outcomes = outcomes
        self.calls = 0
        self.score = None
        self.threshold = 0.5

    @property
    def __name__(self) -> str:
        return self._name

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        self.score = float(outcome)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)


class Mock429Error(RuntimeError):
    status_code = 429

    def __init__(self, retry_after: str | None = None) -> None:
        super().__init__("HTTP 429 rate limit exceeded")
        self.response = type(
            "Response",
            (),
            {
                "status_code": 429,
                "headers": {"Retry-After": retry_after} if retry_after else {},
            },
        )()


def _create_sample_question() -> EvaluationQuestion:
    return EvaluationQuestion(
        id="Q_TEST",
        query="What is the architecture of RAG?",
        reference_answer="RAG combines dense retrieval with generative models.",
        gold_evidence=[GoldEvidence(text="RAG architecture text", element_id="text_1")],
    )


def test_groq_rate_limit_is_detected():
    assert is_groq_rate_limit_error(Mock429Error("12")) is True


def test_answer_evaluator_waits_before_one_controlled_retry():
    metric = MockAnswerMetric("faithfulness", [Mock429Error("12"), 0.9])
    sleeps: list[float] = []
    evaluator = DeepEvalAnswerEvaluator(
        metrics=[metric],
        sleep_fn=sleeps.append,
        rate_limit_wait_seconds=60,
    )

    result = evaluator.evaluate(_create_sample_question(), "Grounded answer", ["Evidence"])

    assert metric.calls == 2
    assert sleeps == [12.0]
    assert result.metrics["faithfulness"].score == pytest.approx(0.9)


def test_answer_evaluator_waits_after_each_429_until_success():
    metric = MockAnswerMetric(
        "faithfulness",
        [Mock429Error(), Mock429Error(), Mock429Error(), 0.8],
    )
    sleeps: list[float] = []
    evaluator = DeepEvalAnswerEvaluator(
        metrics=[metric],
        sleep_fn=sleeps.append,
        rate_limit_wait_seconds=60,
    )

    result = evaluator.evaluate(_create_sample_question(), "Grounded answer", ["Evidence"])

    assert metric.calls == 4
    assert sleeps == [60, 60, 60]
    assert evaluator.rate_limit_events == 3
    assert evaluator.rate_limit_waits == 3
    assert result.metrics["faithfulness"].score == pytest.approx(0.8)


def test_answer_evaluator_starts_next_metric_only_after_previous_succeeds():
    events: list[str] = []
    first = MockAnswerMetric("faithfulness", [Mock429Error("7"), 0.8])
    second = MockAnswerMetric("answer_relevancy", [0.9])

    original_first_measure = first.measure
    original_second_measure = second.measure

    def first_measure(*args, **kwargs):
        events.append("first")
        return original_first_measure(*args, **kwargs)

    def second_measure(*args, **kwargs):
        events.append("second")
        return original_second_measure(*args, **kwargs)

    first.measure = first_measure
    second.measure = second_measure
    evaluator = DeepEvalAnswerEvaluator(
        metrics=[first, second],
        sleep_fn=lambda seconds: events.append(f"wait:{seconds}"),
    )

    result = evaluator.evaluate(_create_sample_question(), "Grounded answer", ["Evidence"])

    assert events == ["first", "wait:7.0", "first", "second"]
    assert list(result.metrics) == ["faithfulness", "answer_relevancy"]


def test_answer_evaluator_records_null_for_genuine_non_rate_limit_failure():
    metric = MockAnswerMetric("faithfulness", [RuntimeError("invalid judge output")])
    sleeps: list[float] = []
    evaluator = DeepEvalAnswerEvaluator(metrics=[metric], sleep_fn=sleeps.append)

    result = evaluator.evaluate(_create_sample_question(), "Grounded answer", ["Evidence"])

    assert metric.calls == 1
    assert sleeps == []
    assert result.metrics["faithfulness"].score is None
    assert "invalid judge output" in (result.metrics["faithfulness"].error or "")


def test_build_llm_test_case_mapping():
    """Verify build_llm_test_case correctly maps query, reference_answer, and contexts."""
    q = _create_sample_question()
    contexts = ["Context snippet 1", "Context snippet 2", "", "   "]

    tc = build_llm_test_case(q, contexts)
    assert isinstance(tc, LLMTestCase)
    assert tc.input == q.query
    assert tc.expected_output == q.reference_answer
    assert tc.retrieval_context == ["Context snippet 1", "Context snippet 2"]


def test_build_llm_test_case_invalid_question():
    """Verify TypeError when invalid question type is passed."""
    with pytest.raises(TypeError, match="EvaluationQuestion"):
        build_llm_test_case("not a question", ["context"])


def test_extract_retrieved_contexts_various_types():
    """Verify context extraction from MultimodalSearchResult, dicts, and strings."""
    elem = MultimodalElement(
        element_id="el_1",
        element_type=ElementType.TEXT,
        content="Element text content from MultimodalElement",
        source_filename="sample_paper.pdf",
        page_number=1,
    )
    search_result = MultimodalSearchResult(element=elem, score=0.9)

    dict_preview = {
        "element_id": "el_2",
        "content_preview": "Preview content from dict",
    }
    dict_full = {
        "element_id": "el_3",
        "content": "Full content from dict",
    }
    plain_str = "Direct context string"

    items = [search_result, dict_preview, dict_full, plain_str, ""]
    extracted = extract_retrieved_contexts(items)

    assert len(extracted) == 4
    assert extracted[0] == "Element text content from MultimodalElement"
    assert extracted[1] == "Preview content from dict"
    assert extracted[2] == "Full content from dict"
    assert extracted[3] == "Direct context string"


def test_normalize_metric_name():
    """Verify metric name normalization into standard keys."""
    assert normalize_metric_name("Contextual Relevancy") == "contextual_relevancy"
    assert normalize_metric_name("contextual_precision_metric") == "contextual_precision"
    assert normalize_metric_name("ContextualRecallMetric") == "contextual_recall"
    assert normalize_metric_name(MockRetrievalMetric(name="Custom Metric")) == "custom"


def test_evaluator_evaluate_contexts_with_mock_metrics():
    """Verify evaluating a single context list with stubbed metrics."""
    q = _create_sample_question()
    m_rel = MockRetrievalMetric("contextual_relevancy", score=0.9, threshold=0.6)
    m_prec = MockRetrievalMetric("contextual_precision", score=0.8, threshold=0.7)
    m_rec = MockRetrievalMetric("contextual_recall", score=0.75, threshold=0.7)

    evaluator = DeepEvalRetrievalEvaluator(metrics=[m_rel, m_prec, m_rec])
    res = evaluator.evaluate_contexts(q, ["Context snippet"], configuration="dense")

    assert isinstance(res, DeepEvalConfigurationResult)
    assert res.question_id == "Q_TEST"
    assert res.configuration == "dense"
    assert len(res.metrics) == 3
    assert res.get_score("contextual_relevancy") == 0.9
    assert res.is_successful("contextual_relevancy") is True
    assert res.get_score("contextual_precision") == 0.8
    assert res.get_score("contextual_recall") == 0.75
    assert len(m_rel.called_with) == 1


def test_evaluator_evaluate_question():
    """Verify evaluating all four configurations on one question."""
    q = _create_sample_question()
    m_rel = MockRetrievalMetric("contextual_relevancy", score=0.8)

    evaluator = DeepEvalRetrievalEvaluator(metrics=[m_rel])
    results_by_config = {
        "dense": ["Dense context snippet"],
        "bm25": ["BM25 context snippet"],
        "hybrid_rrf": ["Hybrid RRF context snippet"],
        "hybrid_reranked": ["Reranked context snippet"],
    }

    q_res = evaluator.evaluate_question(q, results_by_config)
    assert isinstance(q_res, DeepEvalQuestionResult)
    assert q_res.question_id == "Q_TEST"
    assert len(q_res.configurations) == 4
    for cfg in ("dense", "bm25", "hybrid_rrf", "hybrid_reranked"):
        assert cfg in q_res.configurations
        assert q_res.configurations[cfg].get_score("contextual_relevancy") == 0.8


def test_evaluator_evaluate_runs():
    """Verify evaluate_runs with QuestionRetrievalRun objects."""
    q = _create_sample_question()
    m = MockRetrievalMetric("contextual_relevancy", score=0.88)
    evaluator = DeepEvalRetrievalEvaluator(metrics=[m])

    elem = MultimodalElement(
        element_id="el_run",
        element_type=ElementType.TEXT,
        content="Run result text",
        source_filename="sample_paper.pdf",
        page_number=1,
    )
    run = QuestionRetrievalRun(
        question=q,
        results={
            "dense": [MultimodalSearchResult(element=elem, score=0.95)],
            "bm25": [MultimodalSearchResult(element=elem, score=0.90)],
        },
    )

    report = evaluator.evaluate_runs([run])
    assert isinstance(report, DeepEvalBenchmarkReport)
    assert report.total_questions == 1
    assert "dense" in report.aggregate_scores
    assert "bm25" in report.aggregate_scores
    dense_agg = report.aggregate_scores["dense"]["contextual_relevancy"]
    assert dense_agg.mean_score == pytest.approx(0.88)
    assert dense_agg.pass_rate == 1.0


def test_evaluator_evaluate_baseline_json():
    """Verify evaluating the actual baseline_retrieval_results.json across all 8 questions."""
    baseline_path = Path("data/evaluation/baseline_retrieval_results.json")
    if not baseline_path.is_file():
        pytest.skip("baseline_retrieval_results.json not found")

    m_rel = MockRetrievalMetric("contextual_relevancy", score=0.80)
    m_prec = MockRetrievalMetric("contextual_precision", score=0.75)
    m_rec = MockRetrievalMetric("contextual_recall", score=0.70)

    evaluator = DeepEvalRetrievalEvaluator(metrics=[m_rel, m_prec, m_rec])
    report = evaluator.evaluate_baseline_json(baseline_path)

    assert report.total_questions == 8
    expected_configs = ["dense", "bm25", "hybrid_rrf", "hybrid_reranked"]
    for cfg in expected_configs:
        assert cfg in report.configurations
        assert cfg in report.aggregate_scores
        for m_name in ("contextual_relevancy", "contextual_precision", "contextual_recall"):
            assert m_name in report.aggregate_scores[cfg]
            assert report.aggregate_scores[cfg][m_name].total_evaluated == 8

    # Verify summary table formatting
    summary = report.format_summary_table()
    assert "DEEPEVAL RETRIEVAL EVALUATION BENCHMARK — AGGREGATE SUMMARY" in summary
    assert "dense" in summary
    assert "hybrid_reranked" in summary


def test_evaluator_handles_metric_exception_gracefully():
    """Verify evaluator catches metric exceptions without crashing and records the error."""
    q = _create_sample_question()
    failing_metric = MockRetrievalMetric(
        name="contextual_relevancy",
        should_fail=True,
    )
    evaluator = DeepEvalRetrievalEvaluator(metrics=[failing_metric])

    res = evaluator.evaluate_contexts(q, ["Some context"], configuration="dense")
    m_res = res.metrics["contextual_relevancy"]
    assert m_res.score is None
    assert m_res.success is False
    assert "Simulated metric execution failure" in m_res.error


def test_failed_metric_aggregate_is_not_reported_as_zero():
    """All-failure aggregates preserve missing scores instead of becoming valid-looking zeroes."""
    q = _create_sample_question()
    failing_metric = MockRetrievalMetric(
        name="contextual_relevancy",
        should_fail=True,
    )
    evaluator = DeepEvalRetrievalEvaluator(metrics=[failing_metric])

    report = evaluator.evaluate_runs(
        [
            QuestionRetrievalRun(
                question=q,
                results={"dense": []},
                latencies={"dense": 0.0},
            )
        ]
    )
    aggregate = report.aggregate_scores["dense"]["contextual_relevancy"]
    assert aggregate.mean_score is None
    assert aggregate.valid_score_count == 0
    assert aggregate.failed_count == 1
    assert "N/A" in report.format_summary_table()


def test_evaluator_configuration_safety_without_credentials():
    """Verify evaluator reports missing credentials safely without unhandled crashes."""
    # When no model or metrics are provided and OPENAI_API_KEY is not in env,
    # is_configured_for_live_eval returns False or get_metrics raises informative ValueError
    evaluator = DeepEvalRetrievalEvaluator()
    # It should not crash on instantiation
    assert evaluator.model is None
    assert evaluator._metrics is None


def test_extract_retrieved_contexts_precedence():
    """Verify DeepEval extracts full content when available and falls back to preview -> text."""
    # Case 1: Dict has content, content_preview, and text -> extracts full content
    dict_with_all = {
        "content": "Full authoritative content",
        "content_preview": "Short preview...",
        "text": "Fallback text",
    }
    assert extract_retrieved_contexts([dict_with_all]) == ["Full authoritative content"]

    # Case 2: Dict has content_preview and text (no content) -> extracts preview
    dict_preview_only = {
        "content": None,
        "content_preview": "Short preview without full content...",
        "text": "Fallback text",
    }
    assert extract_retrieved_contexts([dict_preview_only]) == ["Short preview without full content..."]

    # Case 3: Dict has only text -> extracts text
    dict_text_only = {
        "text": "Only text provided",
    }
    assert extract_retrieved_contexts([dict_text_only]) == ["Only text provided"]


def test_baseline_json_preserves_full_content():
    """Verify baseline_retrieval_results.json preserves full content alongside content_preview."""
    import json
    baseline_path = Path("data/evaluation/baseline_retrieval_results.json")
    assert baseline_path.is_file()

    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for q in data["questions"]:
        for cfg in ("dense", "bm25", "hybrid_rrf", "hybrid_reranked"):
            items = q["results"][cfg]
            assert len(items) > 0
            for item in items:
                assert "content" in item
                assert "content_preview" in item
                assert len(item["content"]) >= len(item["content_preview"])

    # Specifically check Q1 chunk_10 and chunk_14 full content
    q1 = next(q for q in data["questions"] if q["id"] == "Q1")
    dense_items = {it["element_id"]: it for it in q1["results"]["dense"]}
    assert "sample_paper.pdf#chunk_10" in dense_items
    assert "sample_paper.pdf#chunk_14" in dense_items
    assert "AI for Security" in dense_items["sample_paper.pdf#chunk_10"]["content"]
    assert "Security for AI" in dense_items["sample_paper.pdf#chunk_14"]["content"]
