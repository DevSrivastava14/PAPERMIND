"""DeepEval semantic retrieval evaluation layer for multimodal RAG.

Evaluates retrieved contexts against evaluation questions using DeepEval's
context metrics:
1. ContextualRelevancyMetric: Evaluates signal-to-noise ratio / relevancy of retrieved
   context to the question query.
2. ContextualPrecisionMetric: Evaluates whether relevant context chunks are ranked higher.
3. ContextualRecallMetric: Evaluates whether the retrieved context covers the reference answer.

Designed to be modular and decoupled:
- Operates on both saved baseline JSON files and live QuestionRetrievalRun objects.
- Preserves question IDs and retrieval configuration names in all outputs.
- Supports dependency injection of custom LLM models/providers via `model` parameter,
  avoiding hardcoded credentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from deepeval.metrics import (
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from app.evaluation.models import EvaluationDataset, EvaluationQuestion
from app.evaluation.retrieval_dataset import get_retrieval_evaluation_dataset
from app.evaluation.runner import QuestionRetrievalRun, RetrievalConfiguration
from app.indexing.multimodal_models import MultimodalSearchResult


import re

DEFAULT_METRIC_NAMES: List[str] = [
    "contextual_relevancy",
    "contextual_precision",
    "contextual_recall",
]


def normalize_metric_name(name_or_metric: Union[str, BaseMetric, Any]) -> str:
    """Normalizes a metric name or instance into a standard snake_case key."""
    if isinstance(name_or_metric, str):
        raw = name_or_metric
    elif hasattr(name_or_metric, "__name__"):
        raw = getattr(name_or_metric, "__name__")
    else:
        raw = type(name_or_metric).__name__

    # Convert CamelCase to snake_case
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", raw)
    normalized = (
        s1.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    if normalized.endswith("_metric"):
        normalized = normalized[:-7]
    return normalized



def build_llm_test_case(
    question: EvaluationQuestion,
    retrieved_contexts: Sequence[str],
) -> LLMTestCase:
    """Builds a DeepEval LLMTestCase from an EvaluationQuestion and retrieved contexts.

    Maps:
    - input: question.query
    - expected_output: question.reference_answer
    - retrieval_context: list of string snippets of retrieved chunks
    """
    if not isinstance(question, EvaluationQuestion):
        raise TypeError(
            f"question must be an EvaluationQuestion instance, got {type(question).__name__}"
        )

    contexts = [str(c) for c in retrieved_contexts if c is not None and str(c).strip()]

    return LLMTestCase(
        input=question.query,
        expected_output=question.reference_answer,
        retrieval_context=contexts,
    )


def extract_retrieved_contexts(
    retrieved_items: Sequence[Union[MultimodalSearchResult, Dict[str, Any], str, Any]],
) -> List[str]:
    """Extracts text content snippets from various retrieved item representations.

    Supports:
    - MultimodalSearchResult (via .content or .text_content)
    - Dict items (via "content", "content_preview", or "text")
    - Raw strings
    """
    contexts: List[str] = []
    for item in retrieved_items:
        if isinstance(item, str):
            if item.strip():
                contexts.append(item.strip())
        elif isinstance(item, MultimodalSearchResult):
            content = item.content
            if content is not None and str(content).strip():
                contexts.append(str(content).strip())
            elif item.element and item.element.content:
                contexts.append(str(item.element.content).strip())
        elif isinstance(item, dict):
            # Prefer content -> content_preview -> text
            val = None
            for key in ("content", "content_preview", "text"):
                raw = item.get(key)
                if raw is not None and str(raw).strip():
                    val = str(raw).strip()
                    break
            if val:
                contexts.append(val)

        elif hasattr(item, "content"):
            c = getattr(item, "content")
            if c is not None and str(c).strip():
                contexts.append(str(c).strip())
        elif hasattr(item, "text"):
            t = getattr(item, "text")
            if t is not None and str(t).strip():
                contexts.append(str(t).strip())

    return contexts


@dataclass
class DeepEvalMetricResult:
    """Evaluation result of a single DeepEval metric for one test case.

    Attributes:
        metric_name: Normalized metric name (e.g. 'contextual_relevancy').
        score: Score between 0.0 and 1.0, or None if evaluation failed.
        threshold: Success threshold configured for this metric.
        success: Whether the score satisfied the threshold.
        reason: Optional explanation or diagnostic message generated by DeepEval.
        error: Optional error message if metric measurement raised an exception.
    """

    metric_name: str
    score: Optional[float]
    threshold: float
    success: bool
    reason: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric result to dictionary."""
        return {
            "metric_name": self.metric_name,
            "score": round(self.score, 4) if self.score is not None else None,
            "threshold": self.threshold,
            "success": self.success,
            "reason": self.reason,
            "error": self.error,
        }


@dataclass
class DeepEvalConfigurationResult:
    """DeepEval evaluation results for a single retrieval configuration on one question.

    Attributes:
        question_id: Identifier of the question (e.g. 'Q1').
        configuration: Retrieval configuration name (e.g. 'dense', 'bm25', 'hybrid_rrf', 'hybrid_reranked').
        metrics: Dictionary mapping normalized metric name to DeepEvalMetricResult.
        retrieved_contexts: The list of retrieved context snippets evaluated.
    """

    question_id: str
    configuration: str
    metrics: Dict[str, DeepEvalMetricResult] = field(default_factory=dict)
    retrieved_contexts: List[str] = field(default_factory=list)

    def get_score(self, metric_name: str) -> Optional[float]:
        """Convenience accessor for a specific metric's score."""
        m = self.metrics.get(normalize_metric_name(metric_name))
        return m.score if m else None

    def is_successful(self, metric_name: str) -> bool:
        """Returns True if the specified metric succeeded."""
        m = self.metrics.get(normalize_metric_name(metric_name))
        return m.success if m else False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration result to dictionary."""
        return {
            "question_id": self.question_id,
            "configuration": self.configuration,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "retrieved_contexts_count": len(self.retrieved_contexts),
        }


@dataclass
class DeepEvalQuestionResult:
    """DeepEval evaluation results for one question across all evaluated configurations."""

    question_id: str
    query: str
    reference_answer: str
    configurations: Dict[str, DeepEvalConfigurationResult] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert question result to dictionary."""
        return {
            "question_id": self.question_id,
            "query": self.query,
            "reference_answer": self.reference_answer,
            "configurations": {k: v.to_dict() for k, v in self.configurations.items()},
        }


@dataclass
class DeepEvalMetricAggregate:
    """Aggregated statistics for a specific metric under one retrieval configuration."""

    metric_name: str
    total_evaluated: int
    valid_score_count: int
    failed_count: int
    mean_score: Optional[float]
    pass_count: int
    pass_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric aggregate to dictionary."""
        return {
            "metric_name": self.metric_name,
            "total_evaluated": self.total_evaluated,
            "valid_score_count": self.valid_score_count,
            "failed_count": self.failed_count,
            "mean_score": round(self.mean_score, 4) if self.mean_score is not None else None,
            "pass_count": self.pass_count,
            "pass_rate": round(self.pass_rate, 4),
        }


@dataclass
class DeepEvalBenchmarkReport:
    """Benchmark report of semantic DeepEval metrics across all questions and configurations."""

    total_questions: int
    configurations: List[str]
    evaluated_metrics: List[str]
    question_results: List[DeepEvalQuestionResult]
    aggregate_scores: Dict[str, Dict[str, DeepEvalMetricAggregate]] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark report to dictionary."""
        return {
            "total_questions": self.total_questions,
            "configurations": list(self.configurations),
            "evaluated_metrics": list(self.evaluated_metrics),
            "aggregate_scores": {
                cfg: {m_name: agg.to_dict() for m_name, agg in metrics.items()}
                for cfg, metrics in self.aggregate_scores.items()
            },
            "question_results": [q.to_dict() for q in self.question_results],
        }

    def format_summary_table(self) -> str:
        """Formats a human-readable comparative summary table of DeepEval metrics."""
        lines: List[str] = []
        lines.append("=" * 96)
        lines.append("DEEPEVAL RETRIEVAL EVALUATION BENCHMARK — AGGREGATE SUMMARY")
        lines.append("=" * 96)

        headers = ["Configuration"] + [
            f"{m.replace('_', ' ').title()} (Mean/Pass)" for m in self.evaluated_metrics
        ]
        col_width = 24
        header_line = "".join(f"{h:<{col_width}}" for h in headers)
        separator_line = "".join(f"{'-'*(col_width-2):<{col_width}}" for _ in headers)
        lines.append(header_line)
        lines.append(separator_line)

        for cfg in self.configurations:
            cfg_aggs = self.aggregate_scores.get(cfg, {})
            row = [f"{cfg:<{col_width}}"]
            for m in self.evaluated_metrics:
                agg = cfg_aggs.get(m)
                if agg:
                    if agg.mean_score is None:
                        val_str = f"N/A ({agg.failed_count} failed)"
                    else:
                        val_str = f"{agg.mean_score:.4f} ({agg.pass_rate*100:.1f}%)"
                else:
                    val_str = "N/A"
                row.append(f"{val_str:<{col_width}}")
            lines.append("".join(row))

        lines.append("=" * 96)
        return "\n".join(lines)


def resolve_evaluation_model(
    model: Optional[Union[str, DeepEvalBaseLLM]] = None,
) -> Optional[Union[str, DeepEvalBaseLLM]]:
    """Resolves the evaluation model, automatically configuring Groq if GROQ_API_KEY is set."""
    if model is not None:
        return model

    import os
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        try:
            from dotenv import find_dotenv, load_dotenv
            load_dotenv(find_dotenv(usecwd=True))
            groq_api_key = os.getenv("GROQ_API_KEY")
        except Exception:
            pass

    if groq_api_key:
        # Configure DeepEval retry backoff to comfortably accommodate Groq's ~15s token replenishment window
        if "DEEPEVAL_RETRY_INITIAL_SECONDS" not in os.environ:
            os.environ["DEEPEVAL_RETRY_INITIAL_SECONDS"] = "15.0"
        if "DEEPEVAL_RETRY_MAX_ATTEMPTS" not in os.environ:
            os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] = "6"
        if "DEEPEVAL_RETRY_CAP_SECONDS" not in os.environ:
            os.environ["DEEPEVAL_RETRY_CAP_SECONDS"] = "30.0"

        from deepeval.models.llms.openai_model import OpenAIModel
        groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        return OpenAIModel(
            model=groq_model,
            base_url=base_url,
            api_key=groq_api_key,
            cost_per_input_token=0.0,
            cost_per_output_token=0.0,
        )


    return None


def create_retrieval_metrics(
    model: Optional[Union[str, DeepEvalBaseLLM]] = None,
    threshold: float = 0.5,
    include_reason: bool = True,
    metric_names: Optional[Sequence[str]] = None,
    async_mode: bool = False,
) -> List[BaseMetric]:
    """Factory creating DeepEval retrieval context metrics.

    Supported metric names:
    - 'contextual_relevancy' -> ContextualRelevancyMetric
    - 'contextual_precision' -> ContextualPrecisionMetric
    - 'contextual_recall' -> ContextualRecallMetric
    """
    selected_names = (
        [normalize_metric_name(n) for n in metric_names]
        if metric_names is not None
        else DEFAULT_METRIC_NAMES
    )

    resolved_model = resolve_evaluation_model(model)

    metrics: List[BaseMetric] = []
    kwargs: Dict[str, Any] = {
        "threshold": threshold,
        "include_reason": include_reason,
        "async_mode": async_mode,
    }
    if resolved_model is not None:
        kwargs["model"] = resolved_model

    for name in selected_names:
        if name in ("contextual_relevancy", "relevancy"):
            metrics.append(ContextualRelevancyMetric(**kwargs))
        elif name in ("contextual_precision", "precision"):
            metrics.append(ContextualPrecisionMetric(**kwargs))
        elif name in ("contextual_recall", "recall"):
            metrics.append(ContextualRecallMetric(**kwargs))
        else:
            raise ValueError(
                f"Unsupported retrieval evaluation metric: '{name}'. "
                f"Supported: {DEFAULT_METRIC_NAMES}"
            )

    return metrics



class DeepEvalRetrievalEvaluator:
    """Modular DeepEval semantic evaluator for multimodal retrieval.

    Evaluates retrieved contexts across the four retrieval configurations:
    1. Dense FAISS
    2. BM25
    3. Hybrid RRF
    4. Hybrid RRF + Cross-Encoder

    Accepts an injectable model/provider or pre-instantiated metrics to support
    both real LLM scoring and isolated offline/mock testing.
    """

    def __init__(
        self,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        metrics: Optional[Sequence[BaseMetric]] = None,
        threshold: float = 0.5,
        include_reason: bool = True,
        metric_names: Optional[Sequence[str]] = None,
        throttle_seconds: float = 0.0,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.include_reason = include_reason
        self.throttle_seconds = throttle_seconds
        self.metric_names = (
            [normalize_metric_name(n) for n in metric_names]
            if metric_names is not None
            else list(DEFAULT_METRIC_NAMES)
        )
        self._metrics: Optional[List[BaseMetric]] = list(metrics) if metrics is not None else None

    def get_metrics(self) -> List[BaseMetric]:
        """Returns the configured DeepEval metrics, initializing them lazily if needed."""
        if self._metrics is None:
            try:
                self._metrics = create_retrieval_metrics(
                    model=self.model,
                    threshold=self.threshold,
                    include_reason=self.include_reason,
                    metric_names=self.metric_names,
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to initialize default DeepEval retrieval metrics: {e}. "
                    "Ensure a valid LLM provider model (e.g. OpenAIModel, GeminiModel, "
                    "or custom DeepEvalBaseLLM) is passed via 'model', or provide pre-instantiated "
                    "'metrics', or set the provider API key environment variable."
                ) from e
        return self._metrics

    def is_configured_for_live_eval(self) -> bool:
        """Checks whether the evaluator can initialize and run metrics without missing credentials."""
        try:
            metrics = self.get_metrics()
            return len(metrics) > 0
        except Exception:
            return False

    def evaluate_contexts(
        self,
        question: EvaluationQuestion,
        retrieved_contexts: Sequence[str],
        configuration: str = "unknown",
    ) -> DeepEvalConfigurationResult:
        """Evaluates a single ranked context list for a question using all configured metrics."""
        test_case = build_llm_test_case(
            question=question,
            retrieved_contexts=retrieved_contexts,
        )

        metrics = self.get_metrics()
        results: Dict[str, DeepEvalMetricResult] = {}

        for metric in metrics:
            metric_key = normalize_metric_name(metric)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    metric.measure(test_case)
                    score = getattr(metric, "score", None)
                    reason = getattr(metric, "reason", None)
                    if callable(getattr(metric, "is_successful", None)):
                        success = bool(metric.is_successful())
                    else:
                        success = bool(getattr(metric, "success", False))

                    results[metric_key] = DeepEvalMetricResult(
                        metric_name=metric_key,
                        score=score,
                        threshold=getattr(metric, "threshold", self.threshold),
                        success=success,
                        reason=reason,
                        error=getattr(metric, "error", None),
                    )
                    break
                except Exception as err:
                    err_str = str(err)
                    if "ratelimit" in err_str.lower() and attempt < max_retries - 1:
                        import time
                        time.sleep(15)
                        continue
                    results[metric_key] = DeepEvalMetricResult(
                        metric_name=metric_key,
                        score=None,
                        threshold=getattr(metric, "threshold", self.threshold),
                        success=False,
                        reason=None,
                        error=err_str,
                    )
                    break

            if self.throttle_seconds > 0:
                import time
                time.sleep(self.throttle_seconds)


        return DeepEvalConfigurationResult(
            question_id=question.id,
            configuration=configuration,
            metrics=results,
            retrieved_contexts=list(retrieved_contexts),
        )

    def evaluate_question(
        self,
        question: EvaluationQuestion,
        results_by_config: Dict[str, Sequence[Any]],
    ) -> DeepEvalQuestionResult:
        """Evaluates all retrieval configurations for one EvaluationQuestion."""
        config_results: Dict[str, DeepEvalConfigurationResult] = {}

        for config_name, items in results_by_config.items():
            contexts = extract_retrieved_contexts(items)
            config_result = self.evaluate_contexts(
                question=question,
                retrieved_contexts=contexts,
                configuration=config_name,
            )
            config_results[config_name] = config_result

        return DeepEvalQuestionResult(
            question_id=question.id,
            query=question.query,
            reference_answer=question.reference_answer,
            configurations=config_results,
        )

    def evaluate_question_run(
        self,
        run: QuestionRetrievalRun,
    ) -> DeepEvalQuestionResult:
        """Evaluates a live QuestionRetrievalRun from RetrievalEvaluationRunner."""
        return self.evaluate_question(
            question=run.question,
            results_by_config=run.results,
        )

    def evaluate_runs(
        self,
        runs: Sequence[QuestionRetrievalRun],
    ) -> DeepEvalBenchmarkReport:
        """Evaluates multiple QuestionRetrievalRun instances and aggregates benchmark metrics."""
        question_results = [self.evaluate_question_run(run) for run in runs]
        return self._build_benchmark_report(question_results)

    def evaluate_baseline_json(
        self,
        json_path: Union[str, Path] = "data/evaluation/baseline_retrieval_results.json",
        dataset: Optional[EvaluationDataset] = None,
    ) -> DeepEvalBenchmarkReport:
        """Reads a baseline results JSON file and evaluates all questions across all configurations."""
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"Baseline results file not found at: {path.resolve()}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        eval_dataset = dataset or get_retrieval_evaluation_dataset()
        question_results: List[DeepEvalQuestionResult] = []

        for q_entry in data.get("questions", []):
            qid = q_entry["id"]
            q_obj = eval_dataset.get(qid)
            if q_obj is None:
                continue

            q_res = self.evaluate_question(
                question=q_obj,
                results_by_config=q_entry.get("results", {}),
            )
            question_results.append(q_res)

        return self._build_benchmark_report(question_results)

    def _build_benchmark_report(
        self,
        question_results: Sequence[DeepEvalQuestionResult],
    ) -> DeepEvalBenchmarkReport:
        """Builds aggregated DeepEvalBenchmarkReport from individual question results."""
        if not question_results:
            return DeepEvalBenchmarkReport(
                total_questions=0,
                configurations=[],
                evaluated_metrics=[],
                question_results=[],
                aggregate_scores={},
            )

        # Detect configurations and metrics
        configs: List[str] = []
        metrics_set: List[str] = []

        for qr in question_results:
            for cfg, cfg_res in qr.configurations.items():
                if cfg not in configs:
                    configs.append(cfg)
                for m_name in cfg_res.metrics.keys():
                    if m_name not in metrics_set:
                        metrics_set.append(m_name)

        # Calculate per-configuration, per-metric aggregates
        aggregate_scores: Dict[str, Dict[str, DeepEvalMetricAggregate]] = {}
        for cfg in configs:
            aggregate_scores[cfg] = {}
            for m_name in metrics_set:
                scores: List[float] = []
                pass_count = 0
                total_evaluated = 0
                failed_count = 0

                for qr in question_results:
                    cfg_res = qr.configurations.get(cfg)
                    if not cfg_res:
                        continue
                    m_res = cfg_res.metrics.get(m_name)
                    if not m_res:
                        continue
                    total_evaluated += 1
                    if m_res.score is not None:
                        scores.append(m_res.score)
                    else:
                        failed_count += 1
                    if m_res.success:
                        pass_count += 1

                mean_score = (sum(scores) / len(scores)) if scores else None
                pass_rate = (pass_count / total_evaluated) if total_evaluated > 0 else 0.0

                aggregate_scores[cfg][m_name] = DeepEvalMetricAggregate(
                    metric_name=m_name,
                    total_evaluated=total_evaluated,
                    valid_score_count=len(scores),
                    failed_count=failed_count,
                    mean_score=mean_score,
                    pass_count=pass_count,
                    pass_rate=pass_rate,
                )

        return DeepEvalBenchmarkReport(
            total_questions=len(question_results),
            configurations=configs,
            evaluated_metrics=metrics_set,
            question_results=list(question_results),
            aggregate_scores=aggregate_scores,
        )
