from pathlib import Path
from typing import Any, Dict, List
import pymupdf

from app.ingestion.models import PageContent, ResearchPaperDocument


def extract_pdf_pages(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """Extracts text and metadata page-by-page from a PDF document.

    Args:
        pdf_path: File system path to the PDF document.

    Returns:
        List of dictionaries containing:
            - page_number: 1-based page index (int)
            - text: Extracted plain text content (str)
            - has_text: True if text contains non-whitespace characters (bool)

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is not a valid or readable PDF.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise ValueError(f"Invalid PDF file '{path}': {exc}") from exc

    pages_data: List[Dict[str, Any]] = []
    try:
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
    except Exception as exc:
        raise ValueError(f"Failed to read pages from PDF '{path}': {exc}") from exc
    finally:
        doc.close()

    return pages_data


def parse_research_paper(pdf_path: str | Path) -> ResearchPaperDocument:
    """Parses a research paper PDF into a structured document representation.

    Args:
        pdf_path: File system path to the PDF document.

    Returns:
        A ResearchPaperDocument instance containing the source filename,
        total page count, and page-by-page extracted content.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is not a valid or readable PDF.
    """
    path = Path(pdf_path)
    raw_pages = extract_pdf_pages(path)

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
