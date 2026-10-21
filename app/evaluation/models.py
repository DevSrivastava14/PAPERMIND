"""Data models for evaluation questions, gold evidence, and evaluation datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


class QuestionType(str, Enum):
    """Classification type of evaluation questions."""

    CONCEPTUAL = "conceptual"
    KEYWORD_HEAVY = "keyword-heavy"
    ARCHITECTURE = "architecture / framework structure"
    FIGURE_CAPTION = "figure/caption"
    SPECIFIC_FACTUAL = "specific factual"

    @classmethod
    def from_str(cls, value: str) -> QuestionType:
        """Parse a string to QuestionType case-insensitively."""
        normalized = value.strip().lower()
        for member in cls:
            if member.value.lower() == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(
            f"Unknown question type: '{value}'. Expected one of: "
            f"{[m.value for m in cls]}"
        )


@dataclass
class GoldEvidence:
    """Represents a piece or chunk of gold evidence for an evaluation question.

    Attributes:
        text: Evidence text excerpt or description.
        element_id: Optional concrete element_id of the matching extracted document element.
        page_number: 1-based page number where the evidence appears.
        section: Section name, header, or paragraph descriptor.
        is_distractor: If True, this chunk represents a weak distractor rather than primary evidence.
        metadata: Optional dictionary with extra context or tags.
    """

    text: str
    element_id: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    is_distractor: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("GoldEvidence text cannot be empty")
        if self.page_number is not None and (
            not isinstance(self.page_number, int) or self.page_number < 1
        ):
            raise ValueError(f"page_number must be an integer >= 1, got {self.page_number}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert gold evidence to a dictionary."""
        return {
            "text": self.text,
            "element_id": self.element_id,
            "page_number": self.page_number,
            "section": self.section,
            "is_distractor": self.is_distractor,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GoldEvidence:
        """Create a GoldEvidence instance from a dictionary."""
        return cls(
            text=data["text"],
            element_id=data.get("element_id"),
            page_number=data.get("page_number"),
            section=data.get("section"),
            is_distractor=bool(data.get("is_distractor", False)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class EvaluationQuestion:
    """An evaluation question with reference answer, gold evidence, and evaluation notes.

    Attributes:
        id: Question identifier (e.g. 'Q1', 'Q2').
        query: User question string.
        reference_answer: Ground truth reference answer.
        gold_evidence: List of GoldEvidence chunks.
        question_type: QuestionType categorization.
        special_notes: Optional evaluation or scoring notes.
        metadata: Optional dictionary with extra context.
    """

    id: str
    query: str
    reference_answer: str
    gold_evidence: List[GoldEvidence] = field(default_factory=list)
    question_type: QuestionType = QuestionType.CONCEPTUAL
    special_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("Question id cannot be empty")
        if not self.query or not self.query.strip():
            raise ValueError("Question query cannot be empty")
        if not self.reference_answer or not self.reference_answer.strip():
            raise ValueError("Reference answer cannot be empty")
        if isinstance(self.question_type, str):
            self.question_type = QuestionType.from_str(self.question_type)
        if not self.gold_evidence:
            raise ValueError(f"Question {self.id} must have at least one gold evidence chunk")

    @property
    def has_multiple_evidence(self) -> bool:
        """True if the question has multiple non-distractor gold evidence chunks."""
        return len(self.primary_evidence) > 1

    @property
    def is_multi_chunk(self) -> bool:
        """Alias for has_multiple_evidence (e.g. Q1 and Q4)."""
        return self.has_multiple_evidence

    @property
    def primary_evidence(self) -> List[GoldEvidence]:
        """Returns non-distractor gold evidence chunks."""
        return [e for e in self.gold_evidence if not e.is_distractor]

    @property
    def gold_texts(self) -> List[str]:
        """Returns list of primary gold evidence texts."""
        return [e.text for e in self.primary_evidence]

    @property
    def gold_pages(self) -> List[int]:
        """Returns sorted unique 1-based page numbers containing primary gold evidence."""
        pages = {e.page_number for e in self.primary_evidence if e.page_number is not None}
        return sorted(pages)

    @property
    def gold_element_ids(self) -> List[str]:
        """Returns list of element_ids from primary (non-distractor) gold evidence items."""
        return [e.element_id for e in self.primary_evidence if e.element_id is not None]

    @property
    def unique_gold_element_ids(self) -> List[str]:
        """Returns unique ordered element_ids from primary gold evidence items."""
        seen = set()
        res = []
        for eid in self.gold_element_ids:
            if eid not in seen:
                seen.add(eid)
                res.append(eid)
        return res

    @property
    def requires_separate_caption_scoring(self) -> bool:
        """True if question requires separate figure caption / visual detail evaluation (e.g. Q6)."""
        return self.question_type == QuestionType.FIGURE_CAPTION or (
            self.special_notes is not None and "score this question separately" in self.special_notes
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert question to a serializable dictionary."""
        return {
            "id": self.id,
            "query": self.query,
            "reference_answer": self.reference_answer,
            "question_type": self.question_type.value,
            "gold_evidence": [e.to_dict() for e in self.gold_evidence],
            "gold_element_ids": self.gold_element_ids,
            "special_notes": self.special_notes,
            "metadata": dict(self.metadata),
            "has_multiple_evidence": self.has_multiple_evidence,
            "requires_separate_caption_scoring": self.requires_separate_caption_scoring,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationQuestion:
        """Create an EvaluationQuestion instance from a dictionary."""
        evidence = [GoldEvidence.from_dict(e) for e in data.get("gold_evidence", [])]
        return cls(
            id=data["id"],
            query=data["query"],
            reference_answer=data["reference_answer"],
            gold_evidence=evidence,
            question_type=QuestionType.from_str(data.get("question_type", "conceptual")),
            special_notes=data.get("special_notes"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class EvaluationDataset:
    """Container for a collection of evaluation questions."""

    name: str
    description: str
    questions: List[EvaluationQuestion] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Dataset name cannot be empty")

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self) -> Iterator[EvaluationQuestion]:
        return iter(self.questions)

    def __getitem__(self, index: Union[int, str]) -> EvaluationQuestion:
        if isinstance(index, int):
            return self.questions[index]
        q = self.get(index)
        if q is None:
            raise KeyError(f"Question with id '{index}' not found")
        return q

    def get(self, question_id: str) -> Optional[EvaluationQuestion]:
        """Find question by ID (case-insensitive e.g. 'q1' or 'Q1')."""
        target = question_id.strip().upper()
        for q in self.questions:
            if q.id.strip().upper() == target:
                return q
        return None

    def filter_by_type(
        self, question_type: Union[QuestionType, str]
    ) -> List[EvaluationQuestion]:
        """Filter questions by QuestionType."""
        if isinstance(question_type, str):
            qt = QuestionType.from_str(question_type)
        else:
            qt = question_type
        return [q for q in self.questions if q.question_type == qt]

    @property
    def multi_evidence_questions(self) -> List[EvaluationQuestion]:
        """Questions with multiple gold evidence chunks (e.g. Q1, Q4)."""
        return [q for q in self.questions if q.has_multiple_evidence]

    @property
    def figure_caption_questions(self) -> List[EvaluationQuestion]:
        """Questions focused on figure/caption evaluation (e.g. Q6)."""
        return [q for q in self.questions if q.requires_separate_caption_scoring]

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to a serializable dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "questions": [q.to_dict() for q in self.questions],
        }

    def to_json(self, path: Union[str, Path], indent: int = 2) -> None:
        """Serialize dataset to a JSON file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationDataset:
        """Create an EvaluationDataset from a dictionary."""
        questions = [EvaluationQuestion.from_dict(q) for q in data.get("questions", [])]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            questions=questions,
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> EvaluationDataset:
        """Load an EvaluationDataset from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
