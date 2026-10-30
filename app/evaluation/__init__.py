"""Evaluation module for multimodal retrieval and generation quality."""

from typing import TYPE_CHECKING

from app.evaluation.evaluate_retrieval import (
    build_sample_paper_document,
    format_content_preview,
    format_question_results,
    run_retrieval_baseline,
)
from app.evaluation.models import (
    EvaluationDataset,
    EvaluationQuestion,
    GoldEvidence,
    QuestionType,
)
from app.evaluation.retrieval_dataset import (
    get_retrieval_evaluation_dataset,
    get_retrieval_evaluation_questions,
)
from app.evaluation.runner import (
    ALL_RETRIEVAL_CONFIGURATIONS,
    QuestionRetrievalRun,
    RetrievalConfiguration,
    RetrievalEvaluationRunner,
    create_evaluation_pipeline,
)
from app.evaluation.scorer import (
    ConfigurationAggregateScore,
    ConfigurationScore,
    QuestionScore,
    RetrievalBenchmarkScoreReport,
    score_baseline_json,
    score_evaluation_runs,
    score_question,
    score_question_run,
    score_retrieval_results,
)

if TYPE_CHECKING:
    from app.evaluation.deepeval_evaluator import (
        DeepEvalBenchmarkReport,
        DeepEvalAnswerEvaluator,
        DeepEvalAnswerResult,
        DeepEvalConfigurationResult,
        DeepEvalMetricAggregate,
        DeepEvalMetricResult,
        DeepEvalQuestionResult,
        DeepEvalRetrievalEvaluator,
        build_llm_test_case,
        build_answer_test_case,
        create_answer_metrics,
        create_retrieval_metrics,
        extract_retrieved_contexts,
        normalize_metric_name,
        resolve_evaluation_model,
    )

_DEEPEVAL_SYMBOLS = {
    "DeepEvalBenchmarkReport",
    "DeepEvalAnswerEvaluator",
    "DeepEvalAnswerResult",
    "DeepEvalConfigurationResult",
    "DeepEvalMetricAggregate",
    "DeepEvalMetricResult",
    "DeepEvalQuestionResult",
    "DeepEvalRetrievalEvaluator",
    "build_llm_test_case",
    "build_answer_test_case",
    "create_answer_metrics",
    "create_retrieval_metrics",
    "extract_retrieved_contexts",
    "normalize_metric_name",
    "resolve_evaluation_model",
}


def __getattr__(name: str):
    if name in _DEEPEVAL_SYMBOLS:
        from app.evaluation import deepeval_evaluator
        val = getattr(deepeval_evaluator, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__


__all__ = [
    "QuestionType",
    "GoldEvidence",
    "EvaluationQuestion",
    "EvaluationDataset",
    "get_retrieval_evaluation_questions",
    "get_retrieval_evaluation_dataset",
    "RetrievalConfiguration",
    "ALL_RETRIEVAL_CONFIGURATIONS",
    "QuestionRetrievalRun",
    "RetrievalEvaluationRunner",
    "create_evaluation_pipeline",
    "build_sample_paper_document",
    "format_content_preview",
    "format_question_results",
    "run_retrieval_baseline",
    "ConfigurationScore",
    "QuestionScore",
    "ConfigurationAggregateScore",
    "RetrievalBenchmarkScoreReport",
    "score_retrieval_results",
    "score_question",
    "score_question_run",
    "score_evaluation_runs",
    "score_baseline_json",
    "DeepEvalMetricResult",
    "DeepEvalAnswerResult",
    "DeepEvalConfigurationResult",
    "DeepEvalQuestionResult",
    "DeepEvalMetricAggregate",
    "DeepEvalBenchmarkReport",
    "build_llm_test_case",
    "build_answer_test_case",
    "extract_retrieved_contexts",
    "create_retrieval_metrics",
    "create_answer_metrics",
    "normalize_metric_name",
    "resolve_evaluation_model",
    "DeepEvalRetrievalEvaluator",
    "DeepEvalAnswerEvaluator",
]
