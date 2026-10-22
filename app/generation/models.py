from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

from app.generation.validation import CitationValidationResult
from app.retrieval.models import Citation
from app.retrieval.multimodal_models import MultimodalCitation


@dataclass
class AnswerResult:
    """Structured result of a grounded answer-generation call.

    Attributes:
        answer: The generated answer text, grounded in retrieved evidence.
            If insufficient evidence was available, contains a clear statement
            to that effect and no fabricated claims.
        citations: Structured Citation or MultimodalCitation objects corresponding
            to the evidence blocks used in the prompt. Empty when context was empty.
        query: The original user question that was answered.
        has_context: True when at least one evidence block was supplied to the
            generator; False when the pipeline returned no relevant results.
        model_name: Optional identifier or model name of the generator.
        insufficient_evidence: True when the retrieved context or LLM indicates
            that evidence is insufficient to answer the query.
        validation: Optional CitationValidationResult capturing citation
            and provenance integrity checks.
    """

    answer: str
    citations: Sequence[Union[Citation, MultimodalCitation, Any]]
    query: str
    has_context: bool = True
    model_name: Optional[str] = None
    insufficient_evidence: bool = False
    validation: Optional[CitationValidationResult] = None
