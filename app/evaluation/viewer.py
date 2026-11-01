"""Read-only adapters for persisted evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_EVALUATION_DIR = Path(__file__).resolve().parents[2] / "data" / "evaluation"

ARTIFACT_REGISTRY: dict[str, dict[str, str]] = {
    "baseline": {
        "filename": "baseline_retrieval_results.json",
        "title": "Eight-question retrieval benchmark",
        "kind": "retrieval",
    },
    "day23": {
        "filename": "day23_lightweight_baseline.json",
        "title": "Day 23 lightweight retrieval checkpoint",
        "kind": "retrieval",
    },
    "day25": {
        "filename": "day25_deepeval_results.json",
        "title": "Day 25 DeepEval results",
        "kind": "deepeval",
    },
    "q1_smoke": {
        "filename": "q1_smoke_test_results.json",
        "title": "Q1 cross-configuration smoke test",
        "kind": "deepeval",
    },
}


def list_artifacts() -> list[dict[str, Any]]:
    """Return the fixed artifact catalog without reading arbitrary paths."""
    return [
        {
            "id": artifact_id,
            "filename": entry["filename"],
            "title": entry["title"],
            "kind": entry["kind"],
            "available": (_EVALUATION_DIR / entry["filename"]).is_file(),
        }
        for artifact_id, entry in ARTIFACT_REGISTRY.items()
    ]


def load_artifact(artifact_id: str) -> dict[str, Any]:
    """Load and normalize one allowlisted artifact."""
    entry = ARTIFACT_REGISTRY.get(artifact_id)
    if entry is None:
        raise KeyError(artifact_id)
    path = _EVALUATION_DIR / entry["filename"]
    if not path.is_file():
        raise FileNotFoundError(entry["filename"])
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Evaluation artifact root must be an object")
    if entry["kind"] == "retrieval":
        normalized = _adapt_retrieval(raw, artifact_id)
    else:
        normalized = _adapt_deepeval(raw, artifact_id)
    normalized.update(
        {
            "id": artifact_id,
            "filename": entry["filename"],
            "title": entry["title"],
            "kind": entry["kind"],
            "historical": True,
        }
    )
    return normalized


def _metric(
    value: Any,
    *,
    status: str = "available",
    reason: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Keep value, absence, zero, and failure states distinguishable."""
    return {
        "value": value,
        "status": status,
        "reason": reason,
        "error": error,
    }


def _retrieval_summary(questions: list[dict[str, Any]], configurations: list[str]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for config in configurations:
        total = len(questions)
        partial = complete = reciprocal_sum = 0.0
        for question in questions:
            gold = {
                item.get("element_id")
                for item in question.get("gold_evidence", [])
                if item.get("element_id") and not item.get("is_distractor", False)
            }
            results = question.get("results", {}).get(config, [])
            result_ids = [item.get("element_id") for item in results]
            hits = [idx + 1 for idx, item_id in enumerate(result_ids) if item_id in gold]
            partial += bool(hits)
            complete += bool(gold) and gold.issubset(result_ids)
            reciprocal_sum += 1.0 / min(hits) if hits else 0.0
        summary[config] = {
            "partial_recall": _metric(partial / total if total else None, status="available" if total else "unavailable"),
            "complete_recall": _metric(complete / total if total else None, status="available" if total else "unavailable"),
            "mrr": _metric(reciprocal_sum / total if total else None, status="available" if total else "unavailable"),
        }
    return summary


def _adapt_retrieval(raw: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    metadata = raw.get("metadata", {})
    questions = raw.get("questions", [])
    if artifact_id == "day23":
        configurations = metadata.get("configurations", [])
        normalized_questions = [
            {
                "id": question.get("id"),
                "query": question.get("query"),
                "configurations": question.get("results", {}),
            }
            for question in questions
        ]
        return {
            "metadata": metadata,
            "configurations": configurations,
            "summary": {},
            "questions": normalized_questions,
            "additional_validations": raw.get("additional_validations", {}),
            "modality_coverage": raw.get("modality_coverage", {}),
            "incomplete": False,
        }

    configurations = metadata.get("configurations", [])
    summary = _retrieval_summary(questions, configurations)
    normalized_questions = [
        {
            "id": question.get("id"),
            "query": question.get("query"),
            "question_type": question.get("question_type"),
            "latencies_ms": question.get("latencies_ms", {}),
            "configurations": question.get("results", {}),
        }
        for question in questions
    ]
    return {
        "metadata": metadata,
        "configurations": configurations,
        "summary": summary,
        "questions": normalized_questions,
        "incomplete": False,
    }


def _adapt_deepeval(raw: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    if artifact_id == "q1_smoke":
        configurations = list(raw.get("configurations", {}))
        questions = [
            {
                "id": "Q1",
                "query": raw.get("query"),
                "configurations": raw.get("configurations", {}),
            }
        ]
        return {
            "metadata": {
                "provider": raw.get("provider"),
                "duration_seconds": raw.get("duration_seconds"),
                "all_configurations_completed": raw.get("all_configurations_completed"),
            },
            "configurations": configurations,
            "summary": {},
            "questions": questions,
            "incomplete": False,
        }

    configurations: list[str] = []
    questions = []
    for case in raw.get("cases", []):
        case_configs = {}
        for metric_name, metric in case.get("metrics", {}).items():
            if metric_name not in configurations:
                configurations.append(metric_name)
            case_configs[metric_name] = _metric(
                metric.get("score"),
                status="available" if metric.get("score") is not None else "failed",
                reason=metric.get("reason"),
                error=metric.get("error"),
            )
        questions.append(
            {
                "id": case.get("id"),
                "query": case.get("query"),
                "status": case.get("status"),
                "citation_count": case.get("citation_count"),
                "visual_result_count": case.get("visual_result_count"),
                "configurations": case_configs,
            }
        )
    return {
        "metadata": {
            "generation_model": raw.get("generation_model"),
            "evaluation_model": raw.get("evaluation_model"),
            "api_failures": raw.get("api_failures"),
            "evaluation_rate_limit_events": raw.get("evaluation_rate_limit_events"),
            "evaluation_rate_limit_waits": raw.get("evaluation_rate_limit_waits"),
            "evaluation_metric_attempts": raw.get("evaluation_metric_attempts"),
        },
        "configurations": configurations,
        "summary": {},
        "questions": questions,
        "incomplete": any(
            metric.get("status") == "failed"
            for question in questions
            for metric in question["configurations"].values()
        ),
    }
