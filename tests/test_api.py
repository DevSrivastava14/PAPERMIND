import json
import logging
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from app.api.main import _sanitize_message, app, generate_document_id
from app.multimodal.representation import ElementType
from app.retrieval.multimodal_models import MultimodalCitation, MultimodalRetrievedContext


client = TestClient(app)


def test_query_returns_grounded_context_and_citations() -> None:
    context = MultimodalRetrievedContext(
        formatted_context="[1] [Text] Source: paper.pdf (Page 1)\nEvidence",
        citations=[
            MultimodalCitation(
                citation_label="[1]",
                element_id="paper.pdf#chunk_0",
                element_type=ElementType.TEXT,
                source_filename="paper.pdf",
                page_number=1,
                score=0.9,
            )
        ],
        search_results=[],
    )
    with patch("app.api.main.graph.invoke", return_value={"status": "success", "context": context}) as invoke:
        response = client.post(
            "/query",
            json={"pdf_path": "data/sample_paper.pdf", "query": "What is AI?", "top_k": 2},
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "grounded_context": context.formatted_context,
        "citations": [
            {
                "citation_label": "[1]",
                "element_id": "paper.pdf#chunk_0",
                "element_type": "text",
                "source_filename": "paper.pdf",
                "page_number": 1,
                "score": 0.9,
            }
        ],
        "error": None,
    }
    invoke.assert_called_once()
    assert "x-request-id" in response.headers
    assert len(response.headers["x-request-id"]) > 0


def test_query_returns_clear_graph_error() -> None:
    with patch(
        "app.api.main.graph.invoke",
        return_value={"status": "error", "error": "Ingestion failed: missing.pdf"},
    ):
        response = client.post(
            "/query",
            json={"pdf_path": "missing.pdf", "query": "What is AI?"},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Ingestion failed: missing.pdf"
    assert "x-request-id" in response.headers
    assert len(response.headers["x-request-id"]) > 0


def test_query_rejects_whitespace_query() -> None:
    response = client.post(
        "/query",
        json={"pdf_path": "data/sample_paper.pdf", "query": "   \n\t  "},
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(
        "query cannot be empty or whitespace-only" in err.get("msg", "")
        for err in errors
    )
    assert "x-request-id" in response.headers
    assert len(response.headers["x-request-id"]) > 0


def test_query_rejects_empty_query() -> None:
    response = client.post(
        "/query",
        json={"pdf_path": "data/sample_paper.pdf", "query": ""},
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("at least 1 character" in err.get("msg", "") for err in errors)
    assert "x-request-id" in response.headers


def test_query_rejects_unsupported_file_extension() -> None:
    response = client.post(
        "/query",
        json={"pdf_path": "data/paper.docx", "query": "What is AI?"},
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("only .pdf files are supported" in err.get("msg", "").lower() for err in errors)
    assert "x-request-id" in response.headers


def test_query_rejects_path_traversal() -> None:
    response = client.post(
        "/query",
        json={"pdf_path": "../../secret.pdf", "query": "What is AI?"},
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(
        "Access denied: path escapes allowed document directory" in err.get("msg", "")
        for err in errors
    )
    # Ensure full host filesystem path is not exposed in error payload
    response_text = str(response.json())
    assert "MULTIMODAL_RAG_PROJECT" not in response_text
    assert "x-request-id" in response.headers


def test_query_rejects_unsafe_absolute_path() -> None:
    response = client.post(
        "/query",
        json={"pdf_path": "C:/Windows/System32/drivers/etc/hosts.pdf", "query": "What is AI?"},
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(
        "Access denied: path escapes allowed document directory" in err.get("msg", "")
        for err in errors
    )
    assert "x-request-id" in response.headers


def test_query_preserves_custom_client_request_id() -> None:
    custom_id = "test-req-custom-999"
    with patch("app.api.main.graph.invoke", return_value={"status": "success", "context": MultimodalRetrievedContext(formatted_context="ctx", citations=[], search_results=[])}):
        response = client.post(
            "/query",
            headers={"X-Request-ID": custom_id},
            json={"pdf_path": "data/sample_paper.pdf", "query": "What is AI?", "top_k": 3},
        )
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == custom_id


def test_query_emits_structured_start_and_completion_logs(caplog: pytest.LogCaptureFixture) -> None:
    context = MultimodalRetrievedContext(
        formatted_context="Evidence context",
        citations=[],
        search_results=[],
    )
    secret_query = "What is the secret research topic?"
    with caplog.at_level(logging.INFO, logger="papermind.api"):
        with patch("app.api.main.graph.invoke", return_value={"status": "success", "context": context}):
            response = client.post(
                "/query",
                json={"pdf_path": "data/sample_paper.pdf", "query": secret_query, "top_k": 2},
            )

    assert response.status_code == 200
    req_id = response.headers["x-request-id"]

    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "papermind.api"]
    start_events = [r for r in records if r.get("event") == "request_start"]
    completion_events = [r for r in records if r.get("event") == "request_completion"]

    expected_doc_id = generate_document_id("data/sample_paper.pdf")
    assert len(start_events) == 1
    assert start_events[0]["request_id"] == req_id
    assert start_events[0]["document_id"] == expected_doc_id
    assert start_events[0]["query_length"] == len(secret_query)
    assert start_events[0]["top_k"] == 2

    assert len(completion_events) == 1
    comp = completion_events[0]
    assert comp["request_id"] == req_id
    assert comp["document_id"] == expected_doc_id
    assert comp["query_length"] == len(secret_query)
    assert comp["top_k"] == 2
    assert "retrieval_config" in comp
    assert comp["retrieval_config"]["top_k"] == 2
    assert "candidate_count" in comp
    assert "result_count" in comp
    assert "top_results" in comp
    assert "llm_latency_ms" in comp
    assert "total_latency_ms" in comp
    assert comp["total_latency_ms"] >= 0.0

    # Verify query text is never logged
    for r in caplog.records:
        assert secret_query not in r.getMessage()


def test_query_emits_structured_failure_log_on_error(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="papermind.api"):
        with patch(
            "app.api.main.graph.invoke",
            return_value={"status": "error", "error": "Ingestion failed: missing.pdf"},
        ):
            response = client.post(
                "/query",
                json={"pdf_path": "missing.pdf", "query": "What is AI?"},
            )

    assert response.status_code == 422
    req_id = response.headers["x-request-id"]

    failure_records = [
        json.loads(r.getMessage())
        for r in caplog.records
        if r.name == "papermind.api" and json.loads(r.getMessage()).get("event") == "request_failure"
    ]
    assert len(failure_records) >= 1
    event = failure_records[-1]
    assert event["request_id"] == req_id
    assert event["query_length"] == len("What is AI?")
    assert event["error_category"] == "ingestion_error"
    assert "missing.pdf" in event["error_message"]
    assert "total_latency_ms" in event


def test_sanitize_message_redacts_keys_and_paths() -> None:
    raw = "Failed with gsk_secretApiKey12345678 at C:\\Users\\Dev\\workspace\\data\\file.pdf"
    cleaned = _sanitize_message(raw)
    assert "gsk_secretApiKey12345678" not in cleaned
    assert "[REDACTED_API_KEY]" in cleaned
    assert "C:\\Users\\Dev" not in cleaned
    assert "[PATH]" in cleaned


def test_generate_document_id_is_deterministic_and_safe() -> None:
    from pathlib import Path

    doc_id1 = generate_document_id("data/sample_paper.pdf")
    doc_id2 = generate_document_id("./data/sample_paper.pdf")
    doc_id3 = generate_document_id(Path("data/sample_paper.pdf").resolve())

    assert doc_id1 == doc_id2 == doc_id3
    assert doc_id1.startswith("doc-")
    assert len(doc_id1) == 20
    # Ensure raw directory / user path parts are not exposed
    assert "data" not in doc_id1
    assert "sample_paper" not in doc_id1
    assert "Users" not in doc_id1
    assert "MULTIMODAL_RAG_PROJECT" not in doc_id1


def test_query_passes_request_and_document_id_to_graph() -> None:
    context = MultimodalRetrievedContext(
        formatted_context="ctx",
        citations=[],
        search_results=[],
    )
    with patch("app.api.main.graph.invoke", return_value={"status": "success", "context": context}) as mock_invoke:
        response = client.post(
            "/query",
            json={"pdf_path": "data/sample_paper.pdf", "query": "What is AI?", "top_k": 3},
        )

    assert response.status_code == 200
    mock_invoke.assert_called_once()
    invoked_state = mock_invoke.call_args[0][0]

    expected_doc_id = generate_document_id("data/sample_paper.pdf")
    assert invoked_state["request_id"] == response.headers["x-request-id"]
    assert invoked_state["document_id"] == expected_doc_id
    assert invoked_state["pdf_path"] == "data/sample_paper.pdf"
    assert invoked_state["query"] == "What is AI?"
    assert invoked_state["top_k"] == 3



def test_query_completion_emits_full_metadata_and_safe_summaries(caplog: pytest.LogCaptureFixture) -> None:
    citation = MultimodalCitation(
        citation_label="[1]",
        element_id="paper.pdf#chunk_0",
        element_type=ElementType.TEXT,
        source_filename="paper.pdf",
        page_number=1,
        score=0.9123,
    )
    context = MultimodalRetrievedContext(
        formatted_context="[1] [Text] SECRET EVIDENCE TEXT",
        citations=[citation],
        search_results=[],
        candidate_count=8,
    )

    class DummyReranker:
        pass

    pipeline = type("Pipeline", (), {"reranker": DummyReranker()})()
    state = {
        "status": "success",
        "context": context,
        "pipeline": pipeline,
        "candidate_count": 8,
        "llm_latency_ms": 145.67,
    }

    with caplog.at_level(logging.INFO, logger="papermind.api"):
        with patch("app.api.main.graph.invoke", return_value=state):
            response = client.post(
                "/query",
                json={"pdf_path": "data/sample_paper.pdf", "query": "SECRET_QUERY_TEXT", "top_k": 5},
            )

    assert response.status_code == 200
    comp_events = [
        json.loads(r.getMessage())
        for r in caplog.records
        if r.name == "papermind.api" and json.loads(r.getMessage()).get("event") == "request_completion"
    ]
    assert len(comp_events) == 1
    event = comp_events[0]
    assert event["candidate_count"] == 8
    assert event["result_count"] == 1
    assert event["llm_latency_ms"] == 145.67
    assert event["retrieval_config"] == {"top_k": 5, "reranker": "DummyReranker"}
    assert event["top_results"] == [
        {
            "rank": 1,
            "element_id": "paper.pdf#chunk_0",
            "element_type": "text",
            "score": 0.9123,
        }
    ]

    # Verify no leaks of query or evidence
    for r in caplog.records:
        msg = r.getMessage()
        assert "SECRET_QUERY_TEXT" not in msg
        assert "SECRET EVIDENCE TEXT" not in msg


def test_health_endpoint_returns_healthy_status() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    assert "x-request-id" in response.headers
    assert len(response.headers["x-request-id"]) > 0


def test_health_endpoint_does_not_execute_graph_or_models() -> None:
    with patch("app.api.main.graph.invoke") as mock_invoke:
        response = client.get("/health")
        assert response.status_code == 200
        mock_invoke.assert_not_called()


def test_api_runtime_import_path_does_not_require_deepeval() -> None:
    """Verifies that importing the API runtime path does not require DeepEval."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "sys.modules['deepeval'] = None\n"
        "import app.api.main\n"
        "assert hasattr(app.api.main, 'app')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"API import failed without DeepEval: {result.stderr}"


