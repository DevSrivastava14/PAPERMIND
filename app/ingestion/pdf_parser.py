import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pymupdf

from app.ingestion.models import PageContent, ResearchPaperDocument

DEFAULT_MAX_PDF_SIZE_MB: float = 50.0


def get_max_pdf_size_bytes() -> int:
    """Returns the configured maximum allowed PDF file size in bytes."""
    raw = os.getenv("PAPER_MIND_MAX_PDF_SIZE_MB")
    if raw is not None:
        try:
            val = float(raw)
            if val > 0:
                return int(val * 1024 * 1024)
        except (ValueError, TypeError):
            pass
    return int(DEFAULT_MAX_PDF_SIZE_MB * 1024 * 1024)


def extract_pdf_pages(
    pdf_path: str | Path,
    max_size_bytes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extracts text and metadata page-by-page from a PDF document.

    Args:
        pdf_path: File system path to the PDF document.
        max_size_bytes: Optional maximum allowed file size in bytes. Defaults
            to the configured system limit (50 MB or PAPER_MIND_MAX_PDF_SIZE_MB).

    Returns:
        List of dictionaries containing:
            - page_number: 1-based page index (int)
            - text: Extracted plain text content (str)
            - has_text: True if text contains non-whitespace characters (bool)

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is not a valid or readable PDF, contains no pages,
            has an unsupported format, or exceeds the maximum allowed file size.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Unsupported file format '{path.suffix}': only .pdf files are supported"
        )

    limit_bytes = (
        max_size_bytes if max_size_bytes is not None else get_max_pdf_size_bytes()
    )
    file_size = path.stat().st_size
    if file_size > limit_bytes:
        limit_mb = limit_bytes / (1024 * 1024)
        file_size_mb = file_size / (1024 * 1024)
        raise ValueError(
            f"PDF file size ({file_size_mb:.2f} MB) exceeds maximum allowed limit of {limit_mb:.2f} MB"
        )

    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise ValueError(f"Invalid PDF file '{path}': {exc}") from exc

    pages_data: List[Dict[str, Any]] = []
    try:
        if len(doc) == 0:
            raise ValueError(f"PDF contains no pages: '{path}'")

        for page_idx, page in enumerate(doc, start=1):
            text = page.get_text()
            has_text = bool(text and text.strip())
            pages_data.append(
                {
                    "page_number": page_idx,
                    "text": text,
                    "has_text": has_text,
                }
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to read pages from PDF '{path}': {exc}") from exc
    finally:
        doc.close()

    return pages_data


def parse_research_paper(
    pdf_path: str | Path,
    max_size_bytes: Optional[int] = None,
) -> ResearchPaperDocument:
    """Parses a research paper PDF into a structured document representation.

    Args:
        pdf_path: File system path to the PDF document.
        max_size_bytes: Optional maximum allowed file size in bytes.

    Returns:
        A ResearchPaperDocument instance containing the source filename,
        total page count, and page-by-page extracted content.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is not a valid or readable PDF, contains no pages,
            has an unsupported format, or exceeds the maximum allowed file size.
    """
    path = Path(pdf_path)
    raw_pages = extract_pdf_pages(path, max_size_bytes=max_size_bytes)

    pages = [
        PageContent(
            page_number=page["page_number"],
            text=page["text"],
            has_text=page["has_text"],
        )
        for page in raw_pages
    ]

    return ResearchPaperDocument(
        filename=path.name,
        total_pages=len(pages),
        pages=pages,
    )
