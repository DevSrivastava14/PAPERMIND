from __future__ import annotations

from typing import Any, List, Optional, Union

from app.generation.base import BaseGenerator
from app.generation.models import AnswerResult
from app.generation.prompt_builder import build_grounded_prompt
from app.generation.validation import is_insufficient_evidence, validate_answer_citations
from app.retrieval.models import RetrievedContext
from app.retrieval.multimodal_models import MultimodalRetrievedContext

# Returned verbatim when the pipeline produced no evidence for a query.
_INSUFFICIENT_EVIDENCE_ANSWER = (
    "I cannot answer this question based on the provided documents. "
    "No relevant evidence was retrieved."
)


class AnswerGenerator:
    """Orchestrates grounded answer generation from retrieved evidence.

    Accepts any BaseGenerator implementation so the LLM provider can be
    swapped without modifying retrieval or orchestration logic.
    """

    def __init__(self, generator: BaseGenerator) -> None:
        if not isinstance(generator, BaseGenerator):
            raise TypeError("generator must be an instance of BaseGenerator")
        self.generator = generator

    def generate_from_context(
        self,
        query: str,
        context: Union[RetrievedContext, MultimodalRetrievedContext, Any],
    ) -> AnswerResult:
        """Generates a grounded answer from the pre-assembled retrieval context.

        If ``context`` is empty (no evidence blocks), returns a clear
        insufficient-evidence response immediately without invoking the
        generator. No answer is fabricated.

        Args:
            query: The user's question.
            context: The RetrievedContext or MultimodalRetrievedContext produced
                by the retrieval pipeline.

        Returns:
            An AnswerResult containing the answer text, citations, original
            query, context availability flag, insufficient_evidence status,
            model name, and citation validation result.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        citations = getattr(context, "citations", None) if context is not None else None
        formatted_context = (
            getattr(context, "formatted_context", "") if context is not None else ""
        )

        # Guard: no evidence — do not invoke the model.
        if not citations or not str(formatted_context).strip():
            validation = validate_answer_citations(
                answer=_INSUFFICIENT_EVIDENCE_ANSWER,
                citations=[],
                is_insufficient=True,
            )
            return AnswerResult(
                answer=_INSUFFICIENT_EVIDENCE_ANSWER,
                citations=[],
                query=query,
                has_context=False,
                insufficient_evidence=True,
                model_name=self.generator.model_name,
                validation=validation,
            )

        prompt = build_grounded_prompt(query=query, context=context)
        answer_text = self.generator.generate(prompt)

        is_insufficient = is_insufficient_evidence(answer_text)
        validation = validate_answer_citations(
            answer=answer_text,
            citations=citations,
            is_insufficient=is_insufficient,
        )

        return AnswerResult(
            answer=answer_text,
            citations=list(citations),
            query=query,
            has_context=True,
            insufficient_evidence=is_insufficient,
            model_name=self.generator.model_name,
            validation=validation,
        )


def build_answer_generator(generator: BaseGenerator) -> AnswerGenerator:
    """Convenience factory for AnswerGenerator."""
    return AnswerGenerator(generator=generator)
