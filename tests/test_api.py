from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.main import app
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


def test_query_rejects_empty_query() -> None:
    response = client.post(
        "/query",
        json={"pdf_path": "data/sample_paper.pdf", "query": ""},
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("at least 1 character" in err.get("msg", "") for err in errors)


def test_query_rejects_unsupported_file_extension() -> None:
    response = client.post(
        "/query",
        json={"pdf_path": "data/paper.docx", "query": "What is AI?"},
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("only .pdf files are supported" in err.get("msg", "").lower() for err in errors)


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



