from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def _pdf_bytes(text: str = "Uploaded document") -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    payload = document.tobytes()
    document.close()
    return payload


def test_upload_valid_pdf_and_list_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PAPER_MIND_UPLOAD_ROOT", str(tmp_path / "uploads"))

    response = client.post(
        "/documents",
        files={"file": ("Research Paper.PDF", _pdf_bytes(), "application/pdf")},
    )

    assert response.status_code == 201
    metadata = response.json()
    assert metadata["filename"] == "Research Paper.PDF"
    assert metadata["size_bytes"] > 0
    assert metadata["status"] == "uploaded"
    assert metadata["document_id"].startswith("doc-")
    assert "path" not in metadata

    listed = client.get("/documents")
    assert listed.status_code == 200
    assert listed.json()["documents"] == [metadata]
    stored = list((tmp_path / "uploads").glob("doc-*.pdf"))
    assert len(stored) == 1


def test_upload_duplicate_original_filenames_are_distinct(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PAPER_MIND_UPLOAD_ROOT", str(tmp_path / "uploads"))

    first = client.post("/documents", files={"file": ("paper.pdf", _pdf_bytes("one"))})
    second = client.post("/documents", files={"file": ("paper.pdf", _pdf_bytes("two"))})

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["document_id"] != second.json()["document_id"]
    assert len(client.get("/documents").json()["documents"]) == 2


def test_upload_sanitizes_unsafe_filename_without_exposing_path(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PAPER_MIND_UPLOAD_ROOT", str(tmp_path / "uploads"))

    response = client.post(
        "/documents",
        files={"file": ("../../secret\\research.pdf", _pdf_bytes())},
    )

    assert response.status_code == 201
    assert response.json()["filename"] == "research.pdf"
    assert str(tmp_path) not in response.text


def test_upload_rejects_wrong_extension_and_signature(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PAPER_MIND_UPLOAD_ROOT", str(tmp_path / "uploads"))

    wrong_extension = client.post(
        "/documents",
        files={"file": ("notes.txt", b"%PDF-1.7\nnot really", "text/plain")},
    )
    invalid_signature = client.post(
        "/documents",
        files={"file": ("notes.pdf", b"not a pdf", "application/pdf")},
    )

    assert wrong_extension.status_code == 400
    assert invalid_signature.status_code == 400
    assert list((tmp_path / "uploads").glob(".upload-*")) == []
    assert list((tmp_path / "uploads").glob("doc-*.pdf")) == []


def test_upload_rejects_malformed_and_zero_page_pdfs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PAPER_MIND_UPLOAD_ROOT", str(tmp_path / "uploads"))
    malformed = client.post(
        "/documents",
        files={"file": ("broken.pdf", b"%PDF-1.7\nbroken", "application/pdf")},
    )
    zero_page = client.post(
        "/documents",
        files={"file": ("empty.pdf", b"%PDF-1.7\n", "application/pdf")},
    )

    assert malformed.status_code == 400
    assert zero_page.status_code == 400
    assert list((tmp_path / "uploads").glob(".upload-*")) == []


def test_upload_rejects_oversized_pdf_and_cleans_temporary_file(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PAPER_MIND_UPLOAD_ROOT", str(tmp_path / "uploads"))
    monkeypatch.setenv("PAPER_MIND_MAX_PDF_SIZE_MB", "0.0001")

    response = client.post(
        "/documents",
        files={"file": ("large.pdf", _pdf_bytes("large") + b"x" * 1000)},
    )

    assert response.status_code == 400
    assert "maximum allowed size" in response.json()["detail"].lower()
    assert list((tmp_path / "uploads").glob(".upload-*")) == []
    assert list((tmp_path / "uploads").glob("doc-*.pdf")) == []
