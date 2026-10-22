"""Run the small Day 25 PaperMind generation and DeepEval evaluation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import find_dotenv, load_dotenv

from app.evaluation import (
    DeepEvalAnswerEvaluator,
    EvaluationQuestion,
    GoldEvidence,
    create_answer_metrics,
    extract_retrieved_contexts,
    get_retrieval_evaluation_dataset,
    normalize_metric_name,
)
from app.generation.base import BaseGenerator
from app.orchestration.langgraph_graph import build_papermind_graph


def _load_environment() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)


def _insufficient_question() -> EvaluationQuestion:
    return EvaluationQuestion(
        id="INSUFFICIENT",
        query="What is the exact launch date of the first commercial quantum computer?",
        reference_answer="The provided paper does not contain this information.",
        gold_evidence=[
            GoldEvidence(
                text="No evidence for the requested quantum-computing launch date.",
                metadata={"evaluation_only": True},
            )
        ],
        special_notes="Insufficient-evidence behavior; contextual metrics are not meaningful.",
    )


class _PersistedAnswerGenerator(BaseGenerator):
    """Replays a persisted answer while allowing retrieval context reconstruction."""

    def __init__(self, answer: str) -> None:
        self._answer = answer

    @property
    def model_name(self) -> str:
        return "persisted-answer"

    def generate(self, prompt: str) -> str:
        return self._answer


def main() -> None:
    _load_environment()
    generation_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    evaluation_model = os.getenv("GROQ_EVALUATION_MODEL", "openai/gpt-oss-20b")
    if generation_model == evaluation_model:
        raise RuntimeError("GROQ_EVALUATION_MODEL must differ from GROQ_MODEL.")

    output_path = Path("data/evaluation/day25_deepeval_results.json")
    persisted = json.loads(output_path.read_text(encoding="utf-8")) if output_path.is_file() else {}
    persisted_cases = {case["id"]: case for case in persisted.get("cases", [])}
    dataset = get_retrieval_evaluation_dataset()
    cases = [
        (dataset["Q1"], True),
        (dataset["Q6"], True),
        (_insufficient_question(), False),
    ]
    graph = build_papermind_graph()
    output: Dict[str, Any] = {
        "generation_model": generation_model,
        "evaluation_model": evaluation_model,
        "cases": persisted.get("cases", []),
        "api_failures": [],
        "evaluation_rate_limit_events": 0,
        "evaluation_rate_limit_waits": 0,
        "evaluation_metric_attempts": 0,
    }
    evaluation_started = time.monotonic()

    for question, include_context_metrics in cases:
        case_result = persisted_cases.get(question.id)
        if case_result is None or not case_result.get("answer"):
            raise RuntimeError(
                f"Cannot resume {question.id}: persisted answer is missing."
            )
        try:
            result = graph.invoke(
                {
                    "pdf_path": "data/sample_paper.pdf",
                    "query": question.query,
                    "top_k": 3,
                    "generator": _PersistedAnswerGenerator(case_result["answer"]),
                }
            )
            context = result.get("context")
            answer = case_result["answer"]
            contexts = (
                extract_retrieved_contexts(context.search_results)
                if context is not None
                else []
            )
            configured_metrics = create_answer_metrics(
                include_context_metrics=include_context_metrics
            )
            valid_metrics = {
                name
                for name, metric in case_result.get("metrics", {}).items()
                if metric.get("score") is not None
            }
            pending_metrics = [
                metric
                for metric in configured_metrics
                if normalize_metric_name(metric) not in valid_metrics
            ]
            case_result["status"] = result.get("status")
            case_result["metrics"] = case_result.get("metrics", {})
            case_result["rate_limit_events"] = 0
            case_result["rate_limit_waits"] = 0
            case_result["metric_attempts"] = 0
            for metric in pending_metrics:
                evaluator = DeepEvalAnswerEvaluator(
                    metrics=[metric],
                    include_context_metrics=include_context_metrics,
                )
                evaluation = evaluator.evaluate(question, answer, contexts)
                case_result["rate_limit_events"] += evaluator.rate_limit_events
                case_result["rate_limit_waits"] += evaluator.rate_limit_waits
                case_result["metric_attempts"] += evaluator.metric_attempts
                case_result["metrics"].update(
                    {
                        name: metric_result.to_dict()
                        for name, metric_result in evaluation.metrics.items()
                    }
                )
                output["cases"] = [
                    case
                    for case in output["cases"]
                    if case.get("id") != question.id
                ]
                output["cases"].append(case_result)
                output_path.write_text(
                    json.dumps(output, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as exc:
            output["api_failures"].append(
                {"case": question.id, "error": f"{type(exc).__name__}: {exc}"}
            )

    output["api_failures"] = [
        {
            "case": case["id"],
            "metric": metric_name,
            "error": metric.get("error"),
        }
        for case in output["cases"]
        for metric_name, metric in case.get("metrics", {}).items()
        if metric.get("score") is None and metric.get("error")
    ]
    output["evaluation_rate_limit_events"] = sum(
        case.get("rate_limit_events", 0) for case in output["cases"]
    )
    output["evaluation_rate_limit_waits"] = sum(
        case.get("rate_limit_waits", 0) for case in output["cases"]
    )
    output["evaluation_metric_attempts"] = sum(
        case.get("metric_attempts", 0) for case in output["cases"]
    )
    output["evaluation_duration_seconds"] = round(
        time.monotonic() - evaluation_started, 2
    )
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
