from dataclasses import dataclass, field
from typing import List, Optional

from app.retrieval.models import Citation


@dataclass
class AnswerResult:
    """Structured result of a grounded answer-generation call.

    Attributes:
        answer: The generated answer text, grounded in retrieved evidence.
            If insufficient evidence was available, contains a clear statement
            to that effect and no fabricated claims.
        citations: Structured Citation objects corresponding to the evidence
            blocks used in the prompt. Empty when context was empty.
        query: The original user question that was answered.
        has_context: True when at least one evidence block was supplied to the
            generator; False when the pipeline returned no relevant results.
    """

    answer: str
    citations: List[Citation]
    query: str
    has_context: bool = True
    model_name: Optional[str] = None
