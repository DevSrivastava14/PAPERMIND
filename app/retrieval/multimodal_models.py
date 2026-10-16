"""Data models for multimodal retrieval and context assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from app.indexing.multimodal_models import MultimodalSearchResult
from app.multimodal.representation import ElementType, MultimodalElement


@dataclass
class MultimodalCitation:
    """Represents a citation referencing a specific retrieved multimodal element.

    Attributes:
        citation_label: Sequential label identifier for the citation (e.g. "[1]").
        element_id: Unique string identifier of the referenced element.
        element_type: Modality type of the referenced element.
        source_filename: Name of the originating document/PDF file.
        page_number: 1-based page number where the element appeared.
        score: Optional similarity score of the underlying search result.
    """

    citation_label: str
    element_id: str
    element_type: ElementType
    source_filename: str
    page_number: int
    score: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.citation_label:
            raise ValueError("citation_label cannot be empty")
        if not self.element_id:
            raise ValueError("element_id cannot be empty")
        if not self.source_filename:
            raise ValueError("source_filename cannot be empty")
        if not isinstance(self.page_number, int) or self.page_number < 1:
            raise ValueError(
                f"page_number must be an integer >= 1, got {self.page_number}"
            )
        if isinstance(self.element_type, str):
            self.element_type = ElementType(self.element_type)


@dataclass
class MultimodalRetrievedContext:
    """Encapsulates assembled multimodal prompt context, citations, and search results.

    Attributes:
        formatted_context: Grounded multimodal evidence text formatted with citation markers.
        citations: Structured list of citations matching the evidence blocks.
        search_results: Original MultimodalSearchResult objects returned from vector search.
    """

    formatted_context: str
    citations: List[MultimodalCitation] = field(default_factory=list)
    search_results: List[MultimodalSearchResult] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.citations)

    @property
    def elements(self) -> List[MultimodalElement]:
        """Returns the underlying MultimodalElement objects from search results."""
        return [res.element for res in self.search_results]

    @property
    def visual_results(self) -> List[MultimodalSearchResult]:
        """Returns search results that are visual elements (images)."""
        return [
            res
            for res in self.search_results
            if res.element_type == ElementType.IMAGE
        ]

    @property
    def text_results(self) -> List[MultimodalSearchResult]:
        """Returns search results that are text-like (text, caption, table, equation)."""
        return [
            res
            for res in self.search_results
            if res.element_type != ElementType.IMAGE
        ]

    @property
    def related_element_ids(self) -> List[str]:
        """Returns an ordered, deduplicated list of all related element IDs across results."""
        seen = set()
        related: List[str] = []
        for res in self.search_results:
            for rel_id in res.related_element_ids or []:
                if rel_id and rel_id not in seen:
                    seen.add(rel_id)
                    related.append(rel_id)
        return related

    def get_citation_by_label(self, label: str) -> Optional[MultimodalCitation]:
        """Retrieves a citation by its label string (e.g. '[1]')."""
        for cit in self.citations:
            if cit.citation_label == label:
                return cit
        return None

    def get_result_by_element_id(
        self, element_id: str
    ) -> Optional[MultimodalSearchResult]:
        """Retrieves a search result by its element ID string."""
        for res in self.search_results:
            if res.element_id == element_id:
                return res
        return None
