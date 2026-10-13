from typing import List, Optional

from app.generation.base import BaseGenerator
from app.generation.models import AnswerResult
from app.generation.prompt_builder import build_grounded_prompt
from app.retrieval.models import Citation, RetrievedContext

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
        context: RetrievedContext,
    ) -> AnswerResult:
        """Generates a grounded answer from the pre-assembled retrieval context.

        If ``context`` is empty (no evidence blocks), returns a clear
        insufficient-evidence response immediately without invoking the
        generator. No answer is fabricated.

        Args:
            query: The user's question.
            context: The RetrievedContext produced by the retrieval pipeline.

        Returns:
            An AnswerResult containing the answer text, citations, original
            query, context availability flag, and model name.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        # Guard: no evidence — do not invoke the model.
        if not context.citations:
            return AnswerResult(
                answer=_INSUFFICIENT_EVIDENCE_ANSWER,
                citations=[],
                query=query,
                has_context=False,
                model_name=self.generator.model_name,
            )

        prompt = build_grounded_prompt(query=query, context=context)
        answer_text = self.generator.generate(prompt)

        return AnswerResult(
            answer=answer_text,
            citations=list(context.citations),
            query=query,
            has_context=True,
            model_name=self.generator.model_name,
        )


def build_answer_generator(generator: BaseGenerator) -> AnswerGenerator:
    """Convenience factory for AnswerGenerator."""
    return AnswerGenerator(generator=generator)
