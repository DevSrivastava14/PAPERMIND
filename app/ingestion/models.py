from dataclasses import dataclass
from typing import List


@dataclass
class PageContent:
    """Represents the extracted text and metadata of a single PDF page."""

    page_number: int
    text: str
    has_text: bool


@dataclass
class ResearchPaperDocument:
    """Represents an ingested research paper with page-level content."""

    filename: str
    total_pages: int
    pages: List[PageContent]
