from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.documents.cache import DocumentBundle, clear_document_cache, get_or_build_bundle, invalidate_document_cache
from app.documents.manager import _connect, delete_uploaded_document, get_document


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


def _upload(filename: str, text: str) -> dict:
    response = client.post(
        "/documents",
        files={"file": (filename, BytesIO(_pdf_bytes(text)), "application/pdf")},
    )
    assert response.status_code == 201
    return response.json()


def test_delete_uploaded_document_removes_file_metadata_and_cached_index(
    tmp_path: Path, monkeypatch
) -> None:
    upload_root = tmp_path / "uploads"
    monkeypatch.setenv("PAPER_MIND_UPLOAD_ROOT", str(upload_root))
    clear_document_cache()
    kept = tmp_path / "sample_paper.pdf"
    kept.write_bytes(b"sample-paper")
    evaluation = tmp_path / "evaluation" / "day23_lightweight_baseline.json"
    evaluation.parent.mkdir()
    evaluation.write_text("{}", encoding="utf-8")

    removed = _upload("removable.pdf", "remove me")
    remaining = _upload("remaining.pdf", "keep me")
    removed_path = Path(str(get_document(removed["document_id"])["path"]))
    remaining_path = Path(str(get_document(remaining["document_id"])["path"]))
    builds = {"count": 0}

    def builder() -> DocumentBundle:
        builds["count"] += 1
        return DocumentBundle(object(), object())

    get_or_build_bundle(removed_path, removed["document_id"], builder)
    file_existed_at_invalidation: list[bool] = []

    def spy_invalidate(document_id: str) -> None:
        file_existed_at_invalidation.append(removed_path.exists())
        invalidate_document_cache(document_id)

    with patch("app.documents.manager.invalidate_document_cache", side_effect=spy_invalidate):
        response = client.delete(f"/documents/{removed['document_id']}")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": removed["document_id"],
        "filename": "removable.pdf",
        "removed": True,
    }
    assert file_existed_at_invalidation == [False]
    assert not removed_path.exists()
    assert remaining_path.is_file()
    assert kept.read_bytes() == b"sample-paper"
    assert evaluation.read_text(encoding="utf-8") == "{}"
    listed = client.get("/documents").json()["documents"]
    assert [item["document_id"] for item in listed] == [remaining["document_id"]]
    follow_up = client.post(
        "/query",
        json={"document_id": removed["document_id"], "query": "Where did it go?"},
    )
    assert follow_up.status_code == 404
    clear_document_cache()


def test_invalidate_document_cache_drops_only_that_document(tmp_path: Path) -> None:
    clear_document_cache()
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    get_or_build_bundle(first, "doc-a", lambda: DocumentBundle("first", "index-a"))
    get_or_build_bundle(second, "doc-b", lambda: DocumentBundle("second", "index-b"))

    invalidate_document_cache("doc-a")

    calls: list[str] = []

    def rebuild_first() -> DocumentBundle:
        calls.append("first")
        return DocumentBundle("rebuilt", "index-a")

    def rebuild_second() -> DocumentBundle:
        calls.append("second")
        return DocumentBundle("replaced", "index-b")

    assert get_or_build_bundle(first, "doc-a", rebuild_first).document == "rebuilt"
    assert get_or_build_bundle(second, "doc-b", rebuild_second).document == "second"
    assert calls == ["first"]
    clear_document_cache()


def test_delete_rejects_sample_identifiers_and_registry_path_escape(
    tmp_path: Path, monkeypatch
) -> None:
    upload_root = tmp_path / "uploads"
    monkeypatch.setenv("PAPER_MIND_UPLOAD_ROOT", str(upload_root))
    secret = tmp_path / "secret.pdf"
    secret.write_bytes(b"do-not-delete")
    sample = Path("data/sample_paper.pdf")
    evaluation_names = sorted(path.name for path in Path("data/evaluation").glob("*.json"))
    sample_stamp = sample.stat().st_mtime_ns

    for document_id in ("sample_paper.pdf", "doc-not-a-valid-identifier", "doc-" + "g" * 32):
        response = client.delete(f"/documents/{document_id}")
        assert response.status_code == 404

    poisoned_id = "doc-" + "ab" * 16
    with _connect(upload_root) as connection:
        connection.execute(
            """
            INSERT INTO documents
            (document_id, original_filename, storage_filename, size_bytes, uploaded_at, status)
            VALUES (?, 'secret.pdf', '../secret.pdf', 14, '2026-01-01T00:00:00+00:00', 'uploaded')
            """,
            (poisoned_id,),
        )
        connection.commit()

    with pytest.raises(ValueError, match="cannot be removed"):
        delete_uploaded_document(poisoned_id)

    escaped = client.delete(f"/documents/{poisoned_id}")
    assert escaped.status_code == 400
    assert secret.read_bytes() == b"do-not-delete"
    assert get_document(poisoned_id) is not None
    assert sample.stat().st_mtime_ns == sample_stamp
    assert sorted(path.name for path in Path("data/evaluation").glob("*.json")) == evaluation_names


def test_delete_failure_keeps_uploaded_file_registry_and_cache(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PAPER_MIND_UPLOAD_ROOT", str(tmp_path / "uploads"))
    clear_document_cache()
    uploaded = _upload("fragile.pdf", "still here")
    stored = Path(str(get_document(uploaded["document_id"])["path"]))
    original = get_or_build_bundle(
        stored,
        uploaded["document_id"],
        lambda: DocumentBundle("indexed", "bundle"),
    )

    with patch("pathlib.Path.unlink", side_effect=OSError("denied")):
        response = client.delete(f"/documents/{uploaded['document_id']}")

    assert response.status_code == 500
    assert response.json()["detail"] == "The document could not be removed"
    assert stored.is_file()
    assert client.get("/documents").json()["documents"][0]["document_id"] == uploaded["document_id"]
    assert get_or_build_bundle(stored, uploaded["document_id"], lambda: DocumentBundle("rebuilt", "new")) is original
    clear_document_cache()


def test_registry_failure_after_file_deletion_is_retryable(
    tmp_path: Path, monkeypatch
) -> None:
    import sqlite3

    monkeypatch.setenv("PAPER_MIND_UPLOAD_ROOT", str(tmp_path / "uploads"))
    clear_document_cache()
    uploaded = _upload("partial.pdf", "partial delete")
    document_id = uploaded["document_id"]
    stored = Path(str(get_document(document_id)["path"]))
    get_or_build_bundle(stored, document_id, lambda: DocumentBundle("indexed", "bundle"))
    file_existed_at_invalidation: list[bool] = []

    def fail_once(connection: sqlite3.Connection, target_id: str) -> None:
        raise sqlite3.OperationalError("locked")

    def spy_invalidate(target_id: str) -> None:
        file_existed_at_invalidation.append(stored.exists())
        invalidate_document_cache(target_id)

    with patch("app.documents.manager._delete_registry_row", side_effect=fail_once):
        with patch("app.documents.manager.invalidate_document_cache", side_effect=spy_invalidate):
            with pytest.raises(sqlite3.OperationalError):
                delete_uploaded_document(document_id)

    assert file_existed_at_invalidation == [False]
    assert not stored.exists()
    assert get_document(document_id)["document_id"] == document_id

    removed = delete_uploaded_document(document_id)

    assert removed == {
        "document_id": document_id,
        "filename": "partial.pdf",
        "removed": True,
    }
    assert get_document(document_id) is None
    assert client.get("/documents").json()["documents"] == []
    clear_document_cache()
