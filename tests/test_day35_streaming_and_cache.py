from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.main import app
from app.documents.cache import DocumentBundle, clear_document_cache, get_or_build_bundle
from app.generation.base import BaseGenerator
from app.generation.groq_generator import GroqGenerator
from app.multimodal.collection import MultimodalDocument
from app.multimodal.representation import ElementType, MultimodalElement
from app.indexing.multimodal_models import MultimodalSearchResult
from app.retrieval.multimodal_models import MultimodalCitation, MultimodalRetrievedContext
from app.orchestration.langgraph_graph import ingest_document


client = TestClient(app)


class StreamingGenerator(BaseGenerator):
    def __init__(self, chunks, error=None):
        self.chunks = chunks
        self.error = error

    @property
    def model_name(self) -> str:
        return "test-streaming"

    def generate(self, prompt: str) -> str:
        return "".join(self.chunks)

    def generate_stream(self, prompt: str):
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


def _bundle() -> DocumentBundle:
    document = MultimodalDocument(source_filename="paper.pdf", document_id="doc-test")
    pipeline = object()
    return DocumentBundle(document=document, pipeline=pipeline)


def test_cache_reuses_complete_bundle_and_invalidates_fingerprint(tmp_path: Path) -> None:
    clear_document_cache()
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"first")
    calls = 0

    def build() -> DocumentBundle:
        nonlocal calls
        calls += 1
        return _bundle()

    first = get_or_build_bundle(pdf, "doc-test", build)
    second = get_or_build_bundle(pdf, "doc-test", build)
    assert first is second
    assert calls == 1

    pdf.write_bytes(b"changed")
    third = get_or_build_bundle(pdf, "doc-test", build)
    assert third is not second
    assert calls == 2


def test_cache_keeps_different_documents_isolated(tmp_path: Path) -> None:
    clear_document_cache()
    first_path = tmp_path / "first.pdf"
    second_path = tmp_path / "second.pdf"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    calls = []

    def build_first() -> DocumentBundle:
        calls.append("first")
        return _bundle()

    def build_second() -> DocumentBundle:
        calls.append("second")
        return _bundle()

    first = get_or_build_bundle(first_path, "doc-first", build_first)
    second = get_or_build_bundle(second_path, "doc-second", build_second)
    assert first is not second
    assert calls == ["first", "second"]


def test_cache_does_not_duplicate_concurrent_builds(tmp_path: Path) -> None:
    clear_document_cache()
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    calls = 0

    def build() -> DocumentBundle:
        nonlocal calls
        calls += 1
        return _bundle()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _: get_or_build_bundle(pdf, "doc-test", build),
                range(4),
            )
        )
    assert calls == 1
    assert all(result is results[0] for result in results)


def test_cached_graph_ingestion_reuses_document_and_pipeline(tmp_path: Path) -> None:
    clear_document_cache()
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"paper")
    document = MultimodalDocument(source_filename="paper.pdf", document_id="doc-test")
    pipeline = object()
    with (
        patch("app.orchestration.langgraph_graph.build_document", return_value=document) as build_document,
        patch(
            "app.orchestration.langgraph_graph.create_evaluation_pipeline",
            return_value=pipeline,
        ) as build_pipeline,
    ):
        first = ingest_document(
            {
                "pdf_path": str(pdf),
                "document_id": "doc-test",
                "top_k": 5,
                "use_cache": True,
            }
        )
        second = ingest_document(
            {
                "pdf_path": str(pdf),
                "document_id": "doc-test",
                "top_k": 5,
                "use_cache": True,
            }
        )

    assert first["document"] is document
    assert second["pipeline"] is pipeline
    build_document.assert_called_once()
    build_pipeline.assert_called_once()


def _context() -> MultimodalRetrievedContext:
    element = MultimodalElement(
        element_id="paper.pdf#chunk_0",
        element_type=ElementType.TEXT,
        source_filename="paper.pdf",
        page_number=1,
        content="Evidence",
    )
    citation = MultimodalCitation(
        citation_label="[1]",
        element_id=element.element_id,
        element_type=ElementType.TEXT,
        source_filename="paper.pdf",
        page_number=1,
        score=0.8,
    )
    return MultimodalRetrievedContext(
        formatted_context="[1] Evidence",
        citations=[citation],
        search_results=[MultimodalSearchResult(element=element, score=0.8)],
        candidate_count=1,
    )


def test_streaming_endpoint_emits_retrieval_tokens_and_completion() -> None:
    state = {
        "status": "success",
        "context": _context(),
        "document": MultimodalDocument(source_filename="paper.pdf"),
        "retrieval_latency_ms": 1.0,
    }
    with (
        patch("app.api.main.graph.invoke", return_value=state),
        patch("app.api.main.GroqGenerator") as generator_type,
    ):
        generator_type.return_value = StreamingGenerator(["Answer ", " [1]"])
        response = client.post(
            "/query/stream",
            json={"pdf_path": "data/sample_paper.pdf", "query": "Question"},
        )

    assert response.status_code == 200
    assert "event: retrieval" in response.text
    assert response.text.index("event: retrieval") < response.text.index("event: token")
    assert response.text.count("event: token") == 2
    assert "event: complete" in response.text
    assert '"answer": "Answer  [1]"' in response.text


def test_groq_generator_requests_provider_stream() -> None:
    with patch("app.generation.groq_generator.OpenAI") as openai:
        client_instance = MagicMock()
        openai.return_value = client_instance
        chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="one"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=" two"))]),
        ]
        client_instance.chat.completions.create.return_value = iter(chunks)
        generator = GroqGenerator(api_key="test-key")

        assert list(generator.generate_stream("prompt")) == ["one", " two"]
        client_instance.chat.completions.create.assert_called_once_with(
            model=generator.model_name,
            messages=[{"role": "user", "content": "prompt"}],
            temperature=0.0,
            stream=True,
        )


def test_streaming_provider_failure_emits_error_without_completion() -> None:
    state = {
        "status": "success",
        "context": _context(),
        "document": MultimodalDocument(source_filename="paper.pdf"),
    }

    with (
        patch("app.api.main.graph.invoke", return_value=state),
        patch("app.api.main.GroqGenerator") as generator_type,
    ):
        generator_type.return_value = StreamingGenerator(
            ["partial"], RuntimeError("provider unavailable")
        )
        response = client.post(
            "/query/stream",
            json={"pdf_path": "data/sample_paper.pdf", "query": "Question"},
        )

    assert response.status_code == 200
    assert "event: token" in response.text
    assert "event: error" in response.text
    assert "event: complete" not in response.text
