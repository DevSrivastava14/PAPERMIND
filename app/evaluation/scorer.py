"""Deterministic retrieval scoring module for multimodal RAG evaluation.

Evaluates retrieval results using exact element_id matching against verified gold evidence:
1. partial_evidence_recall: 1 if at least one positive gold element_id was retrieved in top-k, else 0
2. complete_evidence_recall: 1 only if every positive gold element_id was retrieved in top-k, else 0
3. gold_rank: 1-based rank of each retrieved gold element_id, or None if absent

Also aggregates totals (partial recall, complete recall, and MRR) per configuration across questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from app.evaluation.models import EvaluationDataset, EvaluationQuestion
from app.evaluation.retrieval_dataset import get_retrieval_evaluation_dataset
from app.evaluation.runner import QuestionRetrievalRun, RetrievalConfiguration
from app.indexing.multimodal_models import MultimodalSearchResult


@dataclass
class ConfigurationScore:
    """Evaluation score for a single question under a specific retrieval configuration.

    Attributes:
        configuration: Name of the configuration (e.g. 'dense', 'bm25', 'hybrid_rrf', 'hybrid_reranked').
        partial_evidence_recall: 1 if >= 1 positive gold element was retrieved, otherwise 0.
        complete_evidence_recall: 1 only if every positive gold element was retrieved, otherwise 0.
        gold_ranks: Mapping from gold element_id to its 1-based rank (or None if absent).
        total_gold_elements: Number of positive gold elements required.
        retrieved_gold_elements: Number of positive gold elements successfully retrieved in top-k.
    """

    configuration: str
    partial_evidence_recall: int
    complete_evidence_recall: int
    gold_ranks: Dict[str, Optional[int]]
    total_gold_elements: int
    retrieved_gold_elements: int

    @property
    def min_gold_rank(self) -> Optional[int]:
        """Lowest (best) 1-based rank among all retrieved gold elements, or None."""
        present_ranks = [r for r in self.gold_ranks.values() if r is not None]
        return min(present_ranks) if present_ranks else None

    @property
    def reciprocal_rank(self) -> float:
        """Reciprocal of the best gold rank, or 0.0 if no gold element was retrieved."""
        best = self.min_gold_rank
        return (1.0 / best) if best is not None else 0.0

    @property
    def hit(self) -> bool:
        """True if at least one gold element was retrieved."""
        return self.partial_evidence_recall == 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration score to dictionary."""
        return {
            "configuration": self.configuration,
            "partial_evidence_recall": self.partial_evidence_recall,
            "complete_evidence_recall": self.complete_evidence_recall,
            "gold_ranks": dict(self.gold_ranks),
            "total_gold_elements": self.total_gold_elements,
            "retrieved_gold_elements": self.retrieved_gold_elements,
            "min_gold_rank": self.min_gold_rank,
            "reciprocal_rank": round(self.reciprocal_rank, 4),
        }


@dataclass
class QuestionScore:
    """Evaluation score for one question across all evaluated configurations."""

    question_id: str
    question_type: str
    query: str
    gold_element_ids: List[str]
    is_multi_gold: bool
    scores: Dict[str, ConfigurationScore] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize question score to dictionary."""
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "query": self.query,
            "gold_element_ids": list(self.gold_element_ids),
            "is_multi_gold": self.is_multi_gold,
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
        }


@dataclass
class ConfigurationAggregateScore:
    """Aggregated retrieval metrics for one retrieval configuration across all questions.

    Attributes:
        configuration: Name of configuration.
        total_questions: Total number of questions evaluated.
        partial_recall_count: Count of questions with >= 1 gold element retrieved.
        partial_recall_rate: Ratio of partial recall hits (0.0 to 1.0).
        complete_recall_count: Count of questions with ALL gold elements retrieved.
        complete_recall_rate: Ratio of complete recall hits (0.0 to 1.0).
        mrr: Mean Reciprocal Rank across all questions.
        multi_gold_total_questions: Count of multi-gold questions evaluated (e.g. Q1, Q4).
        multi_gold_complete_recall_count: Count of multi-gold questions with all gold elements retrieved.
    """

    configuration: str
    total_questions: int
    partial_recall_count: int
    partial_recall_rate: float
    complete_recall_count: int
    complete_recall_rate: float
    mrr: float
    multi_gold_total_questions: int
    multi_gold_complete_recall_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize aggregate metrics to dictionary."""
        return {
            "configuration": self.configuration,
            "total_questions": self.total_questions,
            "partial_recall_count": self.partial_recall_count,
            "partial_recall_rate": round(self.partial_recall_rate, 4),
            "complete_recall_count": self.complete_recall_count,
            "complete_recall_rate": round(self.complete_recall_rate, 4),
            "mrr": round(self.mrr, 4),
            "multi_gold_total_questions": self.multi_gold_total_questions,
            "multi_gold_complete_recall_count": self.multi_gold_complete_recall_count,
        }


@dataclass
class RetrievalBenchmarkScoreReport:
    """Comprehensive benchmark scoring report containing per-question and aggregate results."""

    total_questions: int
    configurations: List[str]
    question_scores: List[QuestionScore]
    aggregate_scores: Dict[str, ConfigurationAggregateScore]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire report to dictionary."""
        return {
            "total_questions": self.total_questions,
            "configurations": list(self.configurations),
            "aggregate_scores": {k: v.to_dict() for k, v in self.aggregate_scores.items()},
            "question_scores": [q.to_dict() for q in self.question_scores],
        }

    def format_summary_table(self) -> str:
        """Formats a clean human-readable summary table of the aggregate metrics."""
        lines: List[str] = []
        lines.append("=" * 82)
        lines.append("RETRIEVAL EVALUATION BENCHMARK — AGGREGATE SUMMARY")
        lines.append("=" * 82)
        lines.append(
            f"{'Configuration':<24} {'Partial Recall':<16} {'Complete Recall':<18} {'Multi-Gold Complete':<20} {'MRR':<8}"
        )
        lines.append(
            f"{'-'*22:<24} {'-'*14:<16} {'-'*16:<18} {'-'*18:<20} {'-'*6:<8}"
        )

        for config in self.configurations:
            agg = self.aggregate_scores.get(config)
            if not agg:
                continue
            part_str = f"{agg.partial_recall_count}/{agg.total_questions} ({agg.partial_recall_rate * 100:.1f}%)"
            comp_str = f"{agg.complete_recall_count}/{agg.total_questions} ({agg.complete_recall_rate * 100:.1f}%)"
            multi_str = f"{agg.multi_gold_complete_recall_count}/{agg.multi_gold_total_questions}"
            mrr_str = f"{agg.mrr:.4f}"
            lines.append(
                f"{config:<24} {part_str:<16} {comp_str:<18} {multi_str:<20} {mrr_str:<8}"
            )

        lines.append("=" * 82)
        return "\n".join(lines)


def score_retrieval_results(
    retrieved_items: Sequence[Union[MultimodalSearchResult, Dict[str, Any], str]],
    gold_element_ids: Sequence[str],
    configuration: str = "unknown",
) -> ConfigurationScore:
    """Calculates deterministic retrieval scores for a ranked list of items against positive gold element IDs.

    Args:
        retrieved_items: Sequence of MultimodalSearchResult, result dicts, or element_id strings.
        gold_element_ids: Sequence of positive gold element IDs (distractors must already be excluded).
        configuration: Name of configuration being scored.

    Returns:
        ConfigurationScore containing partial_evidence_recall, complete_evidence_recall, and gold_ranks.
    """
    # Extract ordered element_id list from retrieved items
    retrieved_ids: List[str] = []
    for item in retrieved_items:
        if isinstance(item, str):
            retrieved_ids.append(item)
        elif hasattr(item, "element_id"):
            retrieved_ids.append(getattr(item, "element_id"))
        elif isinstance(item, dict) and "element_id" in item:
            retrieved_ids.append(str(item["element_id"]))

    # Deduplicate positive gold IDs while preserving order
    unique_golds: List[str] = []
    seen = set()
    for gid in gold_element_ids:
        if gid and gid not in seen:
            seen.add(gid)
            unique_golds.append(gid)

    if not unique_golds:
        return ConfigurationScore(
            configuration=configuration,
            partial_evidence_recall=0,
            complete_evidence_recall=0,
            gold_ranks={},
            total_gold_elements=0,
            retrieved_gold_elements=0,
        )

    # 1-based ranks of gold elements (or None if missing from retrieved set)
    gold_ranks: Dict[str, Optional[int]] = {}
    retrieved_count = 0
    for gid in unique_golds:
        if gid in retrieved_ids:
            rank = retrieved_ids.index(gid) + 1
            gold_ranks[gid] = rank
            retrieved_count += 1
        else:
            gold_ranks[gid] = None

    partial_recall = 1 if retrieved_count > 0 else 0
    complete_recall = 1 if retrieved_count == len(unique_golds) else 0

    return ConfigurationScore(
        configuration=configuration,
        partial_evidence_recall=partial_recall,
        complete_evidence_recall=complete_recall,
        gold_ranks=gold_ranks,
        total_gold_elements=len(unique_golds),
        retrieved_gold_elements=retrieved_count,
    )


def score_question(
    question: EvaluationQuestion,
    results_by_config: Dict[str, Sequence[Union[MultimodalSearchResult, Dict[str, Any], str]]],
) -> QuestionScore:
    """Scores all retrieval configurations for a single EvaluationQuestion."""
    # Positive gold IDs strictly exclude distractors
    gold_ids = question.unique_gold_element_ids
    is_multi_gold = len(gold_ids) > 1

    scores: Dict[str, ConfigurationScore] = {}
    for config_name, items in results_by_config.items():
        score = score_retrieval_results(
            retrieved_items=items,
            gold_element_ids=gold_ids,
            configuration=config_name,
        )
        scores[config_name] = score

    return QuestionScore(
        question_id=question.id,
        question_type=question.question_type.value,
        query=question.query,
        gold_element_ids=gold_ids,
        is_multi_gold=is_multi_gold,
        scores=scores,
    )


def score_question_run(run: QuestionRetrievalRun) -> QuestionScore:
    """Scores a live QuestionRetrievalRun from RetrievalEvaluationRunner."""
    return score_question(
        question=run.question,
        results_by_config=run.results,
    )


def score_evaluation_runs(
    runs: Sequence[QuestionRetrievalRun],
) -> RetrievalBenchmarkScoreReport:
    """Scores a sequence of QuestionRetrievalRun objects and generates aggregate benchmark metrics."""
    question_scores = [score_question_run(r) for r in runs]
    return build_score_report_from_question_scores(question_scores)


def build_score_report_from_question_scores(
    question_scores: Sequence[QuestionScore],
) -> RetrievalBenchmarkScoreReport:
    """Aggregates a sequence of QuestionScore objects into a comprehensive benchmark report."""
    if not question_scores:
        return RetrievalBenchmarkScoreReport(
            total_questions=0,
            configurations=[],
            question_scores=[],
            aggregate_scores={},
        )

    # Detect all configurations
    configs: List[str] = []
    for qs in question_scores:
        for c in qs.scores.keys():
            if c not in configs:
                configs.append(c)

    total_q = len(question_scores)
    multi_gold_total = sum(1 for qs in question_scores if qs.is_multi_gold)

    aggregate_scores: Dict[str, ConfigurationAggregateScore] = {}
    for c in configs:
        part_count = sum(qs.scores[c].partial_evidence_recall for qs in question_scores if c in qs.scores)
        comp_count = sum(qs.scores[c].complete_evidence_recall for qs in question_scores if c in qs.scores)
        mrr_sum = sum(qs.scores[c].reciprocal_rank for qs in question_scores if c in qs.scores)

        multi_gold_comp = sum(
            qs.scores[c].complete_evidence_recall
            for qs in question_scores
            if c in qs.scores and qs.is_multi_gold
        )

        agg = ConfigurationAggregateScore(
            configuration=c,
            total_questions=total_q,
            partial_recall_count=part_count,
            partial_recall_rate=(part_count / total_q) if total_q > 0 else 0.0,
            complete_recall_count=comp_count,
            complete_recall_rate=(comp_count / total_q) if total_q > 0 else 0.0,
            mrr=(mrr_sum / total_q) if total_q > 0 else 0.0,
            multi_gold_total_questions=multi_gold_total,
            multi_gold_complete_recall_count=multi_gold_comp,
        )
        aggregate_scores[c] = agg

    return RetrievalBenchmarkScoreReport(
        total_questions=total_q,
        configurations=configs,
        question_scores=list(question_scores),
        aggregate_scores=aggregate_scores,
    )


def score_baseline_json(
    json_path: Union[str, Path] = "data/evaluation/baseline_retrieval_results.json",
    dataset: Optional[EvaluationDataset] = None,
) -> RetrievalBenchmarkScoreReport:
    """Reads a baseline results JSON file and scores it using the canonical evaluation dataset."""
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Baseline results file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    eval_dataset = dataset or get_retrieval_evaluation_dataset()
    question_scores: List[QuestionScore] = []

    for q_entry in data.get("questions", []):
        qid = q_entry["id"]
        q_obj = eval_dataset.get(qid)
        if q_obj is None:
            continue

        q_score = score_question(
            question=q_obj,
            results_by_config=q_entry.get("results", {}),
        )
        question_scores.append(q_score)

    return build_score_report_from_question_scores(question_scores)
