from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient

from app.api.main import app
from app.documents.manager import get_document, list_documents
from app.evaluation.evaluate_retrieval import build_document
from app.indexing.multimodal_models import MultimodalSearchResult
from app.multimodal.collection import MultimodalDocument
from app.multimodal.representation import ElementType, MultimodalElement
from app.retrieval.multimodal_models import MultimodalCitation, MultimodalRetrievedContext


client = TestClient(app)


def _pdf_bytes(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    payload = document.tobytes()
    document.close()
    return payload


def _context(filename: str) -> MultimodalRetrievedContext:
    element = MultimodalElement(
        element_id=f"{filename}#chunk_0",
        element_type=ElementType.TEXT,
        source_filename=filename,
        page_number=1,
        content=f"Evidence from {filename}",
    )
    result = MultimodalSearchResult(element=element, score=0.9)
    citation = MultimodalCitation(
        citation_label="[1]",
        element_id=element.element_id,
        element_type=ElementType.TEXT,
        source_filename=filename,
        page_number=1,
        score=0.9,
    )
    return MultimodalRetrievedContext(
        formatted_context=f"[1] Source: {filename}\nEvidence",
        citations=[citation],
        search_results=[result],
    )


def test_build_document_accepts_arbitrary_pdf_and_preserves_identity(tmp_path: Path) -> None:
    pdf_path = tmp_path / "uploaded-paper.pdf"
    pdf_path.write_bytes(_pdf_bytes("Document-specific evidence"))

    document = build_document(pdf_path, document_id="doc-test")

    assert isinstance(document, MultimodalDocument)
    assert document.document_id == "doc-test"
    assert document.source_filename == "uploaded-paper.pdf"
    assert document.elements


def test_upload_query_uses_registry_path_and_marks_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PAPER_MIND_UPLOAD_ROOT", str(tmp_path / "uploads"))
    upload = client.post(
        "/documents",
        files={"file": ("paper.pdf", BytesIO(_pdf_bytes("Uploaded evidence")), "application/pdf")},
    )
    assert upload.status_code == 201
    document_id = upload.json()["document_id"]

    with patch(
        "app.api.main.graph.invoke",
        return_value={
            "status": "success",
            "context": _context("paper.pdf"),
            "document": MultimodalDocument(source_filename="paper.pdf", document_id=document_id),
        },
    ) as invoke:
        response = client.post(
            "/query",
            json={"document_id": document_id, "query": "What is the evidence?"},
        )

    assert response.status_code == 200
    assert response.json()["citations"][0]["source_filename"] == "paper.pdf"
    call_state = invoke.call_args.args[0]
    assert call_state["document_id"] == document_id
    assert Path(call_state["pdf_path"]).name == f"{document_id}.pdf"
    assert get_document(document_id)["status"] == "ready"


def test_unknown_document_id_is_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PAPER_MIND_UPLOAD_ROOT", str(tmp_path / "uploads"))

    response = client.post(
        "/query",
        json={"document_id": "doc-missing", "query": "query"},
    )

    assert response.status_code == 404


def test_uploaded_storage_path_cannot_bypass_registry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    upload_root = tmp_path / "uploads"
    monkeypatch.setenv("PAPER_MIND_UPLOAD_ROOT", str(upload_root))
    monkeypatch.setenv("PAPER_MIND_ALLOWED_ROOTS", str(tmp_path))

    response = client.post(
        "/query",
        json={
            "pdf_path": str(upload_root / "doc-any.pdf"),
            "query": "query",
        },
    )

    assert response.status_code == 422
    assert "document_id" in response.json()["detail"][0]["msg"]


def test_failed_ingestion_is_not_marked_ready(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PAPER_MIND_UPLOAD_ROOT", str(tmp_path / "uploads"))
    upload = client.post(
        "/documents",
        files={"file": ("broken.pdf", BytesIO(_pdf_bytes("Evidence")), "application/pdf")},
    )
    document_id = upload.json()["document_id"]

    with patch(
        "app.api.main.graph.invoke",
        return_value={"status": "error", "error": "Indexing failed"},
    ):
        response = client.post(
            "/query",
            json={"document_id": document_id, "query": "query"},
        )

    assert response.status_code == 422
    assert get_document(document_id)["status"] == "failed"


def test_two_document_queries_keep_registry_identity_isolated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PAPER_MIND_UPLOAD_ROOT", str(tmp_path / "uploads"))
    ids = []
    for filename in ("first.pdf", "second.pdf"):
        response = client.post(
            "/documents",
            files={"file": (filename, BytesIO(_pdf_bytes(filename)), "application/pdf")},
        )
        ids.append(response.json()["document_id"])

    with patch(
        "app.api.main.graph.invoke",
        side_effect=[
            {"status": "success", "context": _context("first.pdf")},
            {"status": "success", "context": _context("second.pdf")},
        ],
    ) as invoke:
        first = client.post("/query", json={"document_id": ids[0], "query": "query"})
        second = client.post("/query", json={"document_id": ids[1], "query": "query"})

    assert first.json()["citations"][0]["source_filename"] == "first.pdf"
    assert second.json()["citations"][0]["source_filename"] == "second.pdf"
    assert invoke.call_args_list[0].args[0]["document_id"] == ids[0]
    assert invoke.call_args_list[1].args[0]["document_id"] == ids[1]
    assert list_documents()[0]["status"] == "ready"
    assert list_documents()[1]["status"] == "ready"
