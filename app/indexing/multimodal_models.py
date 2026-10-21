"""Data models for multimodal indexing and similarity search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from app.multimodal.representation import ElementType, MultimodalElement


@dataclass
class MultimodalSearchResult:
    """Represents a similarity search result matching a query in a multimodal index.

    Preserves the full original MultimodalElement while exposing direct accessors
    for score, element_id, element_type, source_filename, page_number, and relationships.

    Attributes:
        element: The matched original MultimodalElement object.
        score: Cosine similarity score between the query and element vector.
        element_id: Unique string identifier of the matched element.
        element_type: Modality type of the element.
        source_filename: Name of the originating document/PDF file.
        page_number: 1-based page number where the element appeared.
        related_element_ids: IDs of related elements (e.g. caption <-> image/table).
    """

    element: MultimodalElement
    score: float
    element_id: Optional[str] = None
    element_type: Optional[ElementType] = None
    source_filename: Optional[str] = None
    page_number: Optional[int] = None
    related_element_ids: Optional[List[str]] = None
    retrieval_score: Optional[float] = None
    cross_encoder_score: Optional[float] = None
    clip_score: Optional[float] = None
    final_score: Optional[float] = None
    raw_cross_encoder_score: Optional[float] = None
    raw_clip_score: Optional[float] = None

    def __post_init__(self) -> None:
        if self.element is None:
            raise ValueError("element cannot be None")
        if not isinstance(self.element, MultimodalElement):
            raise TypeError(
                f"element must be a MultimodalElement instance, got {type(self.element).__name__}"
            )

        if self.element_id is None:
            self.element_id = self.element.element_id
        if self.element_type is None:
            self.element_type = self.element.element_type
        if self.source_filename is None:
            self.source_filename = self.element.source_filename
        if self.page_number is None:
            self.page_number = self.element.page_number
        if self.related_element_ids is None:
            self.related_element_ids = list(self.element.related_element_ids)

    @property
    def content(self) -> Union[str, bytes, Any]:
        """Convenience property accessing the element's raw content."""
        return self.element.content

    @property
    def metadata(self) -> Dict[str, Any]:
        """Convenience property accessing the element's metadata dictionary."""
        return self.element.metadata
