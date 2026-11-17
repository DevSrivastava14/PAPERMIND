"""Secure storage and metadata registry for uploaded PDF documents."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import secrets
import sqlite3
import tempfile
from typing import BinaryIO

import pymupdf

from app.documents.cache import invalidate_document_cache
from app.ingestion.pdf_parser import get_max_pdf_size_bytes, parse_research_paper


DEFAULT_UPLOAD_ROOT = Path("data/uploads")
UPLOAD_CHUNK_SIZE = 1024 * 1024
PDF_SIGNATURE = b"%PDF-"
DOCUMENT_STATUSES = {"uploaded", "processing", "ready", "failed"}
_DOCUMENT_ID_RE = re.compile(r"^doc-[0-9a-f]{32}$")
_UNSAFE_DELETE = "The document cannot be removed"


def get_upload_root() -> Path:
    """Return the configured application-managed upload directory."""
    configured = os.getenv("PAPER_MIND_UPLOAD_ROOT")
    return Path(configured).expanduser() if configured else DEFAULT_UPLOAD_ROOT


def _registry_path(upload_root: Path) -> Path:
    return upload_root / "documents.sqlite3"


def _connect(upload_root: Path) -> sqlite3.Connection:
    upload_root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(_registry_path(upload_root))
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            original_filename TEXT NOT NULL,
            storage_filename TEXT NOT NULL UNIQUE,
            size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
            uploaded_at TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('uploaded', 'processing', 'ready', 'failed'))
        )
        """
    )
    status_sql = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'documents'"
    ).fetchone()[0]
    if "status = 'uploaded'" in status_sql:
        connection.execute("ALTER TABLE documents RENAME TO documents_legacy")
        connection.execute(
            """
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                storage_filename TEXT NOT NULL UNIQUE,
                size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
                uploaded_at TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('uploaded', 'processing', 'ready', 'failed'))
            )
            """
        )
        connection.execute(
            """
            INSERT INTO documents
            (document_id, original_filename, storage_filename, size_bytes, uploaded_at, status)
            SELECT document_id, original_filename, storage_filename, size_bytes, uploaded_at, 'uploaded'
            FROM documents_legacy
            """
        )
        connection.execute("DROP TABLE documents_legacy")
    connection.commit()
    return connection


def list_documents() -> list[dict[str, object]]:
    """Return uploaded-document metadata without filesystem paths."""
    root = get_upload_root()
    with _connect(root) as connection:
        rows = connection.execute(
            """
            SELECT document_id, original_filename AS filename, size_bytes, uploaded_at, status
            FROM documents
            ORDER BY uploaded_at DESC, document_id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_document(document_id: str) -> dict[str, object] | None:
    """Return trusted metadata and the application-managed path for one document."""
    if not document_id or not document_id.strip():
        return None
    root = get_upload_root()
    with _connect(root) as connection:
        row = connection.execute(
            """
            SELECT document_id, original_filename AS filename, storage_filename,
                   size_bytes, uploaded_at, status
            FROM documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["path"] = root / str(result["storage_filename"])
    return result


def _registered_upload_path(root: Path, document_id: str, storage_filename: str) -> Path:
    """Resolve one server-generated upload name and reject anything outside that root."""
    expected_name = f"{document_id}.pdf"
    if storage_filename != expected_name or Path(storage_filename).name != storage_filename:
        raise ValueError(_UNSAFE_DELETE)
    candidate = root / storage_filename
    if candidate.is_symlink():
        raise ValueError(_UNSAFE_DELETE)
    resolved = candidate.resolve()
    if resolved.parent != root or not resolved.is_relative_to(root):
        raise ValueError(_UNSAFE_DELETE)
    if resolved.exists() and not resolved.is_file():
        raise ValueError(_UNSAFE_DELETE)
    return resolved


def _delete_registry_row(connection: sqlite3.Connection, document_id: str) -> None:
    cursor = connection.execute(
        "DELETE FROM documents WHERE document_id = ?",
        (document_id,),
    )
    if cursor.rowcount != 1:
        raise KeyError(document_id)


def delete_uploaded_document(document_id: str) -> dict[str, object]:
    """Remove one uploaded document, its file, and its cached index bundle.

    Only registry records with server-generated ids can be deleted. Sample papers,
    evaluation artifacts, and paths outside the upload root are never touched.
    """
    if not _DOCUMENT_ID_RE.fullmatch(document_id or ""):
        raise KeyError(document_id)
    record = get_document(document_id)
    if record is None:
        raise KeyError(document_id)

    root = get_upload_root().resolve()
    file_path = _registered_upload_path(root, document_id, str(record["storage_filename"]))
    try:
        file_path.unlink(missing_ok=True)
    except OSError:
        raise

    try:
        with _connect(root) as connection:
            _delete_registry_row(connection, document_id)
            connection.commit()
    except Exception:
        if not file_path.exists():
            invalidate_document_cache(document_id)
        raise

    invalidate_document_cache(document_id)
    return {
        "document_id": document_id,
        "filename": record["filename"],
        "removed": True,
    }


def update_document_status(document_id: str, status: str) -> None:
    """Set a registered document's processing status."""
    if status not in DOCUMENT_STATUSES:
        raise ValueError(f"Unsupported document status: {status}")
    root = get_upload_root()
    with _connect(root) as connection:
        cursor = connection.execute(
            "UPDATE documents SET status = ? WHERE document_id = ?",
            (status, document_id),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown document: {document_id}")
        connection.commit()


def _sanitize_filename(filename: str) -> str:
    name = Path(filename.replace("\\", "/")).name
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    if not name or name in {".", ".."}:
        raise ValueError("A valid PDF filename is required")
    return name


def _validate_pdf_file(path: Path, size_bytes: int) -> None:
    if size_bytes > get_max_pdf_size_bytes():
        raise ValueError("The PDF exceeds the maximum allowed size")
    with path.open("rb") as stream:
        if stream.read(len(PDF_SIGNATURE)) != PDF_SIGNATURE:
            raise ValueError("The uploaded file is not a valid PDF")
        payload = path.read_bytes()
    try:
        pdf = pymupdf.open(stream=payload, filetype="pdf")
        if len(pdf) == 0:
            raise ValueError("The uploaded PDF contains no pages")
        pdf.close()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The uploaded file is not a readable PDF") from exc
    try:
        parse_research_paper(path, max_size_bytes=get_max_pdf_size_bytes())
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable PDF") from exc


def _copy_upload(upload_stream: BinaryIO, target: Path) -> int:
    total = 0
    limit = get_max_pdf_size_bytes()
    with target.open("wb") as destination:
        while True:
            chunk = upload_stream.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ValueError("The PDF exceeds the maximum allowed size")
            destination.write(chunk)
    return total


def save_uploaded_pdf(filename: str, upload_stream: BinaryIO) -> dict[str, object]:
    """Validate and atomically persist one uploaded PDF, then register its metadata."""
    original_filename = _sanitize_filename(filename)
    if not original_filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported")

    root = get_upload_root()
    root.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    final_path: Path | None = None
    registered = False
    document_id = f"doc-{secrets.token_hex(16)}"
    storage_filename = f"{document_id}.pdf"

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".upload-",
            suffix=".pdf",
            dir=root,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        size_bytes = _copy_upload(upload_stream, temporary_path)
        _validate_pdf_file(temporary_path, size_bytes)

        final_path = root / storage_filename
        os.replace(temporary_path, final_path)
        temporary_path = None

        uploaded_at = datetime.now(timezone.utc).isoformat()
        try:
            with _connect(root) as connection:
                connection.execute(
                    """
                    INSERT INTO documents
                    (document_id, original_filename, storage_filename, size_bytes, uploaded_at, status)
                    VALUES (?, ?, ?, ?, ?, 'uploaded')
                    """,
                    (
                        document_id,
                        original_filename,
                        storage_filename,
                        size_bytes,
                        uploaded_at,
                    ),
                )
                connection.commit()
                registered = True
        except sqlite3.Error as exc:
            raise ValueError("The uploaded document could not be registered") from exc

        return {
            "document_id": document_id,
            "filename": original_filename,
            "size_bytes": size_bytes,
            "uploaded_at": uploaded_at,
            "status": "uploaded",
        }
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if final_path is not None and not registered:
            final_path.unlink(missing_ok=True)
