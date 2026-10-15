from pathlib import Path
from typing import Any, List, Optional
import pymupdf

from app.multimodal.models import ExtractedTable


class TableExtractor:
    """Extracts structured tables from research PDF pages using PyMuPDF."""

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the table extractor.

        Args:
            **kwargs: Optional keyword arguments passed to PyMuPDF's page.find_tables().
        """
        self.find_tables_kwargs = kwargs

    def extract(self, pdf_path: str | Path) -> List[ExtractedTable]:
        """Extracts all tables from the given PDF document.

        Preserves exact 1-based page provenance and deterministic table indexing.

        Args:
            pdf_path: File system path to the PDF document.

        Returns:
            List of ExtractedTable instances representing tables found.
            Returns an empty list if the PDF contains no tables.

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

        extracted: List[ExtractedTable] = []
        try:
            for page_idx, page in enumerate(doc, start=1):
                try:
                    tabs = page.find_tables(**self.find_tables_kwargs)
                    table_list = tabs.tables if hasattr(tabs, "tables") else list(tabs)
                except Exception:
                    table_list = []

                for tab_idx, tab in enumerate(table_list):
                    table_id = f"{path.name}#page_{page_idx}_table_{tab_idx}"

                    # Extract tabular cells
                    raw_rows: List[Any] = []
                    if hasattr(tab, "extract"):
                        try:
                            raw_rows = tab.extract() or []
                        except Exception:
                            raw_rows = []

                    cleaned_rows = [
                        [str(cell).strip() if cell is not None else "" for cell in row]
                        for row in raw_rows
                    ]

                    # Bounding box
                    bbox = None
                    if hasattr(tab, "bbox") and tab.bbox is not None:
                        try:
                            bbox = tuple(float(x) for x in tab.bbox)
                        except Exception:
                            bbox = None

                    # Content / markdown
                    content = None
                    if hasattr(tab, "to_markdown"):
                        try:
                            md = tab.to_markdown()
                            if md and md.strip():
                                content = md.strip()
                        except Exception:
                            content = None

                    # Column headers
                    headers = None
                    if hasattr(tab, "header") and tab.header is not None:
                        try:
                            if hasattr(tab.header, "names") and tab.header.names:
                                headers = [
                                    str(n).strip() if n is not None else ""
                                    for n in tab.header.names
                                ]
                        except Exception:
                            headers = None

                    extracted.append(
                        ExtractedTable(
                            table_id=table_id,
                            source_filename=path.name,
                            page_number=page_idx,
                            table_index=tab_idx,
                            rows=cleaned_rows,
                            content=content,
                            bbox=bbox,
                            headers=headers,
                        )
                    )
        except Exception as exc:
            raise ValueError(f"Failed to read tables from PDF '{path}': {exc}") from exc
        finally:
            doc.close()

        return extracted


def extract_tables(pdf_path: str | Path, **kwargs: Any) -> List[ExtractedTable]:
    """Convenience function to extract tables from a PDF document.

    Args:
        pdf_path: File system path to the PDF document.
        **kwargs: Optional parameters passed to TableExtractor.

    Returns:
        List of ExtractedTable instances.
    """
    extractor = TableExtractor(**kwargs)
    return extractor.extract(pdf_path)
