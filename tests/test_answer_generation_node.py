from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.generation import (
    AnswerGenerator,
    AnswerResult,
    BaseGenerator,
    CitationValidationResult,
    GroqGenerator,
    build_answer_generator,
    is_insufficient_evidence,
    validate_answer_citations,
)
from app.generation.answer_generator import _INSUFFICIENT_EVIDENCE_ANSWER
from app.multimodal.representation import ElementType
from app.orchestration.langgraph_graph import (
    build_papermind_graph,
    generate_answer,
)
from app.retrieval.multimodal_models import (
    MultimodalCitation,
    MultimodalRetrievedContext,
)


class MockLLM(BaseGenerator):
    """Deterministic in-memory LLM generator for testing."""

    def __init__(self, answer: str = "The model achieves 28.4 BLEU [1].") -> None:
        self._answer = answer
        self.recorded_prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return "mock-groq-qwen"

    def generate(self, prompt: str) -> str:
        self.recorded_prompts.append(prompt)
        return self._answer


@pytest.fixture
def multimodal_context() -> MultimodalRetrievedContext:
    cit1 = MultimodalCitation(
        citation_label="[1]",
        element_id="paper.pdf#chunk_0",
        element_type=ElementType.TEXT,
        source_filename="paper.pdf",
        page_number=1,
        score=0.92,
    )
    cit2 = MultimodalCitation(
        citation_label="[2]",
        element_id="paper.pdf#page_2_img_1",
        element_type=ElementType.IMAGE,
        source_filename="paper.pdf",
        page_number=2,
        score=0.88,
    )
    formatted = (
        "[1] [Text] Source: paper.pdf (Page 1)\nThe model achieves 28.4 BLEU.\n\n"
        "[2] [Image] Source: paper.pdf (Page 2)\nArchitecture diagram of the transformer."
    )
    return MultimodalRetrievedContext(
        formatted_context=formatted,
        citations=[cit1, cit2],
        search_results=[],
    )


# ---------------------------------------------------------------------------
# GroqGenerator tests
# ---------------------------------------------------------------------------

class TestGroqGenerator:
    """Verifies Groq configuration resolution and invocation."""

    def test_groq_generator_reads_env_and_config(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "GROQ_API_KEY": "test-key-123",
                "GROQ_MODEL": "qwen/qwen3.8-27b",
                "GROQ_BASE_URL": "https://api.groq.com/openai/v1",
            },
        ):
            with patch("app.generation.groq_generator.OpenAI") as mock_openai:
                gen = GroqGenerator()
                assert gen.model_name == "qwen/qwen3.8-27b"
                mock_openai.assert_called_once_with(
                    api_key="test-key-123",
                    base_url="https://api.groq.com/openai/v1",
                    timeout=30.0,
                )

    def test_groq_generator_raises_when_no_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("dotenv.find_dotenv", return_value=""):
                with pytest.raises(ValueError, match="GROQ_API_KEY is required"):
                    GroqGenerator()

    def test_groq_generator_generate_calls_chat_completion(self) -> None:
        with patch("app.generation.groq_generator.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.choices = [
                MagicMock(message=MagicMock(content="   Answer from Groq [1]   "))
            ]
            mock_client.chat.completions.create.return_value = mock_resp

            gen = GroqGenerator(api_key="dummy-key", model="qwen/qwen3.8-27b")
            reply = gen.generate("Prompt here")

            assert reply == "Answer from Groq [1]"
            mock_client.chat.completions.create.assert_called_once_with(
                model="qwen/qwen3.8-27b",
                messages=[{"role": "user", "content": "Prompt here"}],
                temperature=0.0,
            )


# ---------------------------------------------------------------------------
# Multimodal Answer Generation & Provenance tests
# ---------------------------------------------------------------------------

class TestAnswerGeneratorMultimodal:
    """Verifies that structured MultimodalRetrievedContext is passed into LLM."""

    def test_multimodal_context_preserves_citations_and_provenance(
        self, multimodal_context: MultimodalRetrievedContext
    ) -> None:
        llm = MockLLM(answer="The system achieves 28.4 BLEU [1] with architecture [2].")
        generator = build_answer_generator(llm)

        result = generator.generate_from_context(
            query="What is the BLEU score?",
            context=multimodal_context,
        )

        assert isinstance(result, AnswerResult)
        assert result.has_context is True
        assert result.insufficient_evidence is False
        assert len(result.citations) == 2

        # Check multimodal provenance preservation
        c1 = result.citations[0]
        assert c1.citation_label == "[1]"
        assert c1.element_id == "paper.pdf#chunk_0"
        assert c1.element_type == ElementType.TEXT
        assert c1.source_filename == "paper.pdf"
        assert c1.page_number == 1
        assert c1.score == 0.92

        c2 = result.citations[1]
        assert c2.citation_label == "[2]"
        assert c2.element_id == "paper.pdf#page_2_img_1"
        assert c2.element_type == ElementType.IMAGE
        assert c2.source_filename == "paper.pdf"
        assert c2.page_number == 2
        assert c2.score == 0.88

        # Verify LLM received formatted context without reconstruction
        assert len(llm.recorded_prompts) == 1
        prompt = llm.recorded_prompts[0]
        assert multimodal_context.formatted_context in prompt
        assert "Question: What is the BLEU score?" in prompt


# ---------------------------------------------------------------------------
# Insufficient Evidence Path tests
# ---------------------------------------------------------------------------

class TestInsufficientEvidencePath:
    """Verifies explicit insufficient-evidence handling."""

    def test_empty_citations_returns_insufficient_evidence_without_llm_call(
        self,
    ) -> None:
        llm = MockLLM()
        generator = build_answer_generator(llm)
        empty_ctx = MultimodalRetrievedContext(
            formatted_context="",
            citations=[],
            search_results=[],
        )

        result = generator.generate_from_context(
            query="What is the result?",
            context=empty_ctx,
        )

        assert result.answer == _INSUFFICIENT_EVIDENCE_ANSWER
        assert result.has_context is False
        assert result.insufficient_evidence is True
        assert result.citations == []
        assert len(llm.recorded_prompts) == 0

    def test_llm_refusal_is_detected_as_insufficient_evidence(
        self, multimodal_context: MultimodalRetrievedContext
    ) -> None:
        refusal_text = (
            "I cannot answer this question based on the provided documents. "
            "The provided excerpts do not mention recipes or cooking."
        )
        llm = MockLLM(answer=refusal_text)
        generator = build_answer_generator(llm)

        result = generator.generate_from_context(
            query="What is the cake recipe?",
            context=multimodal_context,
        )

        assert result.insufficient_evidence is True
        assert result.has_context is True
        assert result.answer == refusal_text

    @pytest.mark.parametrize(
        "phrase",
        [
            "I cannot answer this question based on the provided documents.",
            "There is insufficient evidence to determine the exact date.",
            "The provided excerpts do not contain information on this.",
            "No relevant evidence was retrieved.",
            "The provided documents do not mention quantum entanglement.",
            # Day 25 exact failure wording
            (
                "The provided evidence excerpts do not contain information regarding the launch date "
                "of the first commercial quantum computer. The sources discuss topics such as AI and "
                "security [2], AI alignment with nature-inspired mechanisms [1], and various AI "
                "applications in risk management and information security [3], but none mention quantum computing."
            ),
            # Reasonable refusal variations
            "The provided evidence excerpts do not contain information regarding this topic.",
            "The retrieved context does not contain information on this question.",
            "There is no mention of quantum computing in the available sources.",
            "The provided information does not contain details about the launch date.",
            "The provided evidence does not contain sufficient details to answer.",
        ],
    )
    def test_is_insufficient_evidence_indicators(self, phrase: str) -> None:
        assert is_insufficient_evidence(phrase) is True

    @pytest.mark.parametrize(
        "answer",
        [
            "The model achieves 28.4 BLEU score on WMT 2014 [1].",
            "AI for Security means using AI technologies to solve security problems [1].",
            "Figure 2 shows Weak AI vs Strong AI [1].",
        ],
    )
    def test_sufficient_evidence_answer_not_flagged(self, answer: str) -> None:
        assert is_insufficient_evidence(answer) is False

    @pytest.mark.parametrize(
        "empty_input",
        [
            "",
            "   ",
            "\n\t  \n",
            None,
        ],
    )
    def test_empty_or_invalid_answer_flagged_as_insufficient(
        self, empty_input: str | None
    ) -> None:
        assert is_insufficient_evidence(empty_input) is True

    def test_day25_exact_refusal_detected_in_generator(
        self, multimodal_context: MultimodalRetrievedContext
    ) -> None:
        day25_refusal = (
            "The provided evidence excerpts do not contain information regarding the launch date "
            "of the first commercial quantum computer. The sources discuss topics such as AI and "
            "security [2], AI alignment with nature-inspired mechanisms [1], and various AI "
            "applications in risk management and information security [3], but none mention quantum computing."
        )
        llm = MockLLM(answer=day25_refusal)
        generator = build_answer_generator(llm)

        result = generator.generate_from_context(
            query="What is the exact launch date of the first commercial quantum computer?",
            context=multimodal_context,
        )

        assert result.insufficient_evidence is True
        assert result.has_context is True
        assert result.answer == day25_refusal


# ---------------------------------------------------------------------------
# Lightweight Citation Validation tests
# ---------------------------------------------------------------------------

class TestCitationValidation:
    """Tests lightweight citation and provenance validation."""

    def test_valid_inline_citations_and_provenance(
        self, multimodal_context: MultimodalRetrievedContext
    ) -> None:
        answer = "The transformer model achieves strong results [1] as seen in [2]."
        val = validate_answer_citations(answer, multimodal_context.citations)

        assert val.is_valid is True
        assert val.cited_labels == ["[1]", "[2]"]
        assert val.invalid_labels == []
        assert len(val.errors) == 0

    def test_hallucinated_citation_marker_is_flagged(
        self, multimodal_context: MultimodalRetrievedContext
    ) -> None:
        answer = "According to previous work [1] and speculative theorem [99]."
        val = validate_answer_citations(answer, multimodal_context.citations)

        assert val.is_valid is False
        assert "[99]" in val.invalid_labels
        assert any("[99]" in err for err in val.errors)

    def test_missing_provenance_is_detected(self) -> None:
        broken_cit = Mock()
        broken_cit.citation_label = "[1]"
        broken_cit.source_filename = ""  # missing
        broken_cit.page_number = -1  # invalid
        broken_cit.element_id = ""  # missing
        broken_cit.chunk_id = ""  # missing

        val = validate_answer_citations("Answer [1]", [broken_cit])
        assert val.is_valid is False
        assert any("source_filename" in err for err in val.errors)
        assert any("page_number" in err for err in val.errors)
        assert any("element_id" in err for err in val.errors)


# ---------------------------------------------------------------------------
# LangGraph Integration tests
# ---------------------------------------------------------------------------

class TestLangGraphAnswerGenerationNode:
    """Tests LangGraph answer generation node and full flow integration."""

    def test_generate_answer_node_with_custom_generator(
        self, multimodal_context: MultimodalRetrievedContext
    ) -> None:
        llm = MockLLM(answer="AI for Security uses AI to defend systems [1].")
        state = {
            "query": "Explain AI for Security",
            "context": multimodal_context,
            "generator": llm,
        }

        output = generate_answer(state)

        assert output["status"] == "success"
        assert output["answer"] == "AI for Security uses AI to defend systems [1]."
        assert output["insufficient_evidence"] is False
        assert output["answer_result"] is not None

    def test_graph_flow_runs_end_to_end_with_generator(
        self, multimodal_context: MultimodalRetrievedContext
    ) -> None:
        document = Mock()
        pipeline = Mock()
        pipeline.hybrid_retrieve_reranked_and_assemble.return_value = multimodal_context
        llm = MockLLM(answer="Answer from pipeline [1].")

        with (
            patch(
                "app.orchestration.langgraph_graph.build_sample_paper_document",
                return_value=document,
            ),
            patch(
                "app.orchestration.langgraph_graph.create_evaluation_pipeline",
                return_value=pipeline,
            ),
        ):
            graph = build_papermind_graph()
            result = graph.invoke(
                {
                    "pdf_path": "test.pdf",
                    "query": "What is the answer?",
                    "generator": llm,
                }
            )

        assert result["status"] == "success"
        assert result["answer"] == "Answer from pipeline [1]."
        assert result["answer_result"] is not None
        assert result["context"] is multimodal_context

    def test_graph_flow_converts_llm_timeout_to_controlled_error(
        self, multimodal_context: MultimodalRetrievedContext
    ) -> None:
        document = Mock()
        pipeline = Mock()
        pipeline.hybrid_retrieve_reranked_and_assemble.return_value = multimodal_context

        failing_llm = Mock(spec=BaseGenerator)
        failing_llm.model_name = "mock-groq-qwen"
        failing_llm.generate.side_effect = TimeoutError("Request timed out after 30.0s")

        with (
            patch(
                "app.orchestration.langgraph_graph.build_sample_paper_document",
                return_value=document,
            ),
            patch(
                "app.orchestration.langgraph_graph.create_evaluation_pipeline",
                return_value=pipeline,
            ),
        ):
            graph = build_papermind_graph()
            result = graph.invoke(
                {
                    "pdf_path": "test.pdf",
                    "query": "What is the answer?",
                    "generator": failing_llm,
                }
            )

        assert result["status"] == "error"
        assert result["error"] == "Answer generation failed: Request timed out after 30.0s"

    def test_graph_insufficient_evidence_path(self) -> None:
        document = Mock()
        pipeline = Mock()
        empty_ctx = MultimodalRetrievedContext(
            formatted_context="",
            citations=[],
            search_results=[],
        )
        pipeline.hybrid_retrieve_reranked_and_assemble.return_value = empty_ctx
        llm = MockLLM()

        with (
            patch(
                "app.orchestration.langgraph_graph.build_sample_paper_document",
                return_value=document,
            ),
            patch(
                "app.orchestration.langgraph_graph.create_evaluation_pipeline",
                return_value=pipeline,
            ),
        ):
            graph = build_papermind_graph()
            result = graph.invoke(
                {
                    "pdf_path": "test.pdf",
                    "query": "What is the answer?",
                    "generator": llm,
                }
            )

        assert result["status"] == "success"
        assert result["insufficient_evidence"] is True
        assert result["answer"] == _INSUFFICIENT_EVIDENCE_ANSWER
        assert len(llm.recorded_prompts) == 0

    def test_graph_irrelevant_evidence_triggers_grounded_refusal(
        self, multimodal_context: MultimodalRetrievedContext
    ) -> None:
        document = Mock()
        pipeline = Mock()
        pipeline.hybrid_retrieve_reranked_and_assemble.return_value = multimodal_context
        refusal = "I cannot answer this question based on the provided documents. The provided excerpts do not contain information regarding this topic."
        llm = MockLLM(answer=refusal)

        with (
            patch(
                "app.orchestration.langgraph_graph.build_sample_paper_document",
                return_value=document,
            ),
            patch(
                "app.orchestration.langgraph_graph.create_evaluation_pipeline",
                return_value=pipeline,
            ),
        ):
            graph = build_papermind_graph()
            result = graph.invoke(
                {
                    "pdf_path": "test.pdf",
                    "query": "What is the launch date of quantum computers?",
                    "generator": llm,
                }
            )

        assert result["status"] == "success"
        assert result["insufficient_evidence"] is True
        assert result["answer"] == refusal
        assert len(llm.recorded_prompts) == 1



# ---------------------------------------------------------------------------
# API /query Endpoint tests
# ---------------------------------------------------------------------------

class TestAPIAnswerGeneration:
    """Tests /query endpoint returning grounded answer and citations."""

    def test_query_returns_answer_and_citations(
        self, multimodal_context: MultimodalRetrievedContext
    ) -> None:
        client = TestClient(app)
        state_return = {
            "status": "success",
            "context": multimodal_context,
            "answer": "Grounded answer text [1].",
            "insufficient_evidence": False,
        }
        with patch("app.api.main.graph.invoke", return_value=state_return):
            response = client.post(
                "/query",
                json={
                    "pdf_path": "data/sample_paper.pdf",
                    "query": "What is the result?",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["answer"] == "Grounded answer text [1]."
        assert data["insufficient_evidence"] is False
        assert len(data["citations"]) == 2
        assert data["citations"][0]["element_id"] == "paper.pdf#chunk_0"
        assert data["citations"][1]["element_type"] == "image"
        assert data["grounded_context"] == multimodal_context.formatted_context

    def test_query_returns_insufficient_evidence_response(self) -> None:
        client = TestClient(app)
        empty_ctx = MultimodalRetrievedContext(
            formatted_context="",
            citations=[],
            search_results=[],
        )
        state_return = {
            "status": "success",
            "context": empty_ctx,
            "answer": _INSUFFICIENT_EVIDENCE_ANSWER,
            "insufficient_evidence": True,
        }
        with patch("app.api.main.graph.invoke", return_value=state_return):
            response = client.post(
                "/query",
                json={
                    "pdf_path": "data/sample_paper.pdf",
                    "query": "Random query?",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["answer"] == _INSUFFICIENT_EVIDENCE_ANSWER
        assert data["insufficient_evidence"] is True
        assert data["citations"] == []


    def test_generate_answer_emits_structured_completion_log(
        self, caplog: pytest.LogCaptureFixture, multimodal_context: MultimodalRetrievedContext
    ) -> None:
        import json
        import logging

        llm = MockLLM(answer="The model achieves 28.4 BLEU [1].")
        state = {
            "query": "SECRET_QUESTION_NOT_TO_BE_LOGGED",
            "context": multimodal_context,
            "generator": llm,
            "request_id": "req-gen-001",
            "document_id": "doc-gen-002",
        }

        with caplog.at_level(logging.INFO, logger="papermind.generation"):
            output = generate_answer(state)

        assert output["status"] == "success"
        assert "llm_latency_ms" in output
        assert output["llm_latency_ms"] >= 0.0

        records = [
            json.loads(r.getMessage())
            for r in caplog.records
            if r.name == "papermind.generation" and json.loads(r.getMessage()).get("event") == "generation_completion"
        ]
        assert len(records) == 1
        log = records[0]
        assert log["request_id"] == "req-gen-001"
        assert log["document_id"] == "doc-gen-002"
        assert log["model_name"] == "mock-groq-qwen"
        assert log["insufficient_evidence"] is False
        assert log["citations_count"] >= 1
        assert "llm_latency_ms" in log
        assert log["llm_latency_ms"] >= 0.0

        # Verify prompt and evidence texts are never logged
        for r in caplog.records:
            msg = r.getMessage()
            assert "SECRET_QUESTION_NOT_TO_BE_LOGGED" not in msg
            assert "Architecture diagram of the transformer" not in msg

    def test_generate_answer_emits_structured_failure_log(
        self, caplog: pytest.LogCaptureFixture, multimodal_context: MultimodalRetrievedContext
    ) -> None:
        import json
        import logging

        failing_llm = Mock(spec=BaseGenerator)
        failing_llm.model_name = "mock-groq-qwen"
        failing_llm.generate.side_effect = TimeoutError(
            "Groq request timed out using key gsk_secretLlmKey at C:\\Users\\Dev\\file.py"
        )

        state = {
            "query": "query",
            "context": multimodal_context,
            "generator": failing_llm,
            "request_id": "req-gen-fail",
            "document_id": "doc-gen-fail",
        }

        with caplog.at_level(logging.WARNING, logger="papermind.generation"):
            output = generate_answer(state)

        assert output["status"] == "error"
        assert "Answer generation failed" in output["error"]

        records = [
            json.loads(r.getMessage())
            for r in caplog.records
            if r.name == "papermind.generation" and json.loads(r.getMessage()).get("event") == "generation_failure"
        ]
        assert len(records) == 1
        log = records[0]
        assert log["request_id"] == "req-gen-fail"
        assert log["document_id"] == "doc-gen-fail"
        assert log["error_category"] == "timeout_error"
        assert "llm_latency_ms" in log
        assert log["llm_latency_ms"] >= 0.0
        assert "gsk_secretLlmKey" not in log["error_message"]
        assert "[REDACTED_API_KEY]" in log["error_message"]
        assert "C:\\Users\\Dev" not in log["error_message"]
        assert "[PATH]" in log["error_message"]
