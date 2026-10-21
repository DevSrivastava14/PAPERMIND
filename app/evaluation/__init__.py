"""Evaluation module for multimodal retrieval and generation quality."""

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
from app.evaluation.deepeval_evaluator import (
    DeepEvalBenchmarkReport,
    DeepEvalConfigurationResult,
    DeepEvalMetricAggregate,
    DeepEvalMetricResult,
    DeepEvalQuestionResult,
    DeepEvalRetrievalEvaluator,
    build_llm_test_case,
    create_retrieval_metrics,
    extract_retrieved_contexts,
    normalize_metric_name,
    resolve_evaluation_model,
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
    "DeepEvalConfigurationResult",
    "DeepEvalQuestionResult",
    "DeepEvalMetricAggregate",
    "DeepEvalBenchmarkReport",
    "build_llm_test_case",
    "extract_retrieved_contexts",
    "create_retrieval_metrics",
    "normalize_metric_name",
    "DeepEvalRetrievalEvaluator",
]

