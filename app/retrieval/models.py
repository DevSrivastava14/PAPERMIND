from dataclasses import dataclass
from typing import List, Optional

from app.indexing.models import SearchResult


@dataclass
class Citation:
    """Represents a citation referencing a specific retrieved text chunk.

    Attributes:
        citation_label: Sequential label identifier for the citation (e.g. "[1]").
        source_filename: Name of the original PDF document file.
        page_numbers: 1-based list of page numbers where the text chunk originated.
        chunk_id: Unique string identifier of the referenced chunk.
        score: Optional similarity score of the underlying search result.
    """

    citation_label: str
    source_filename: str
    page_numbers: List[int]
    chunk_id: str
    score: Optional[float] = None


@dataclass
class RetrievedContext:
    """Encapsulates the assembled prompt context, citations, and search results.

    Attributes:
        formatted_context: Grounded evidence text formatted with citation markers.
        citations: Structured list of citations matching the evidence blocks.
        search_results: Original SearchResult objects returned from vector search.
    """

    formatted_context: str
    citations: List[Citation]
    search_results: List[SearchResult]

    def __len__(self) -> int:
        return len(self.citations)
