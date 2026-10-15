"""Unified representation layer for multimodal scientific-paper elements.

This module provides a unified abstraction (`MultimodalElement`) and adapters
for all modalities extracted from scientific papers (text, images, captions,
tables, equations) so downstream embedding, indexing, and retrieval can
operate over a single consistent interface while preserving full provenance and
inter-element relationships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from app.chunking.models import TextChunk
from app.multimodal.models import (
    CaptionMatch,
    ExtractedCaption,
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
)


class ElementType(str, Enum):
    """Explicit supported multimodal element types."""

    TEXT = "text"
    IMAGE = "image"
    CAPTION = "caption"
    TABLE = "table"
    EQUATION = "equation"


ELEMENT_TYPE_TEXT: str = ElementType.TEXT.value
ELEMENT_TYPE_IMAGE: str = ElementType.IMAGE.value
ELEMENT_TYPE_CAPTION: str = ElementType.CAPTION.value
ELEMENT_TYPE_TABLE: str = ElementType.TABLE.value
ELEMENT_TYPE_EQUATION: str = ElementType.EQUATION.value

SUPPORTED_ELEMENT_TYPES = {e.value for e in ElementType}

_TYPE_SYNONYMS = {
    "figure": ElementType.IMAGE,
    "fig": ElementType.IMAGE,
    "img": ElementType.IMAGE,
    "formula": ElementType.EQUATION,
    "math": ElementType.EQUATION,
}


@dataclass
class MultimodalElement:
    """Unified representation of a scientific document element across all modalities.

    Attributes:
        element_id: Unique deterministic identifier for this element.
        element_type: Modality type (text, image, caption, table, equation).
        source_filename: Name of the originating PDF/document file.
        page_number: 1-based page number where the element appeared in the PDF.
        content: Modality-aware content (text string for text/table/equation/caption,
            or binary bytes / path reference for images).
        embedding: Optional dense embedding vector for downstream retrieval.
        related_element_ids: IDs of related elements (e.g. caption <-> figure/table).
        metadata: Modality-specific attributes (bbox, format, rows, headers, etc.).
    """

    element_id: str
    element_type: ElementType
    source_filename: str
    page_number: int
    content: Union[str, bytes, Any]
    embedding: Optional[Union[List[float], np.ndarray]] = None
    related_element_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        element_id: str,
        element_type: Union[ElementType, str],
        source_filename: str,
        page_number: int,
        content: Union[str, bytes, Any],
        embedding: Optional[Union[List[float], np.ndarray]] = None,
        related_element_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        relationships: Optional[List[str]] = None,
        related_ids: Optional[List[str]] = None,
    ) -> None:
        self.element_id = str(element_id).strip() if element_id is not None else ""
        self.source_filename = (
            str(source_filename).strip() if source_filename is not None else ""
        )
        self.page_number = page_number
        self.content = content
        self.embedding = embedding

        # Resolve relationships list
        raw_related = (
            related_element_ids
            if related_element_ids is not None
            else (relationships if relationships is not None else related_ids)
        )
        if raw_related is not None and not isinstance(raw_related, (list, tuple)):
            raise TypeError(
                f"related_element_ids must be a list, got {type(raw_related).__name__}"
            )
        self.related_element_ids = (
            [str(r).strip() for r in raw_related if r is not None and str(r).strip()]
            if raw_related is not None
            else []
        )

        self.metadata = dict(metadata) if metadata is not None else {}

        # Resolve element_type
        self.element_type = self._normalize_type(element_type)

        self._validate()

    @staticmethod
    def _normalize_type(val: Union[ElementType, str]) -> ElementType:
        if isinstance(val, ElementType):
            return val
        if isinstance(val, str):
            clean = val.strip().lower()
            if clean in _TYPE_SYNONYMS:
                return _TYPE_SYNONYMS[clean]
            if clean in SUPPORTED_ELEMENT_TYPES:
                return ElementType(clean)
            raise ValueError(
                f"Invalid element_type '{val}'. Supported types: {sorted(SUPPORTED_ELEMENT_TYPES)}"
            )
        raise ValueError(
            f"element_type must be an ElementType or string, got {type(val).__name__}"
        )

    def _validate(self) -> None:
        if not self.element_id:
            raise ValueError("element_id cannot be empty")
        if not self.source_filename:
            raise ValueError("source_filename cannot be empty")
        if (
            not isinstance(self.page_number, int)
            or isinstance(self.page_number, bool)
            or self.page_number < 1
        ):
            raise ValueError(
                f"page_number must be an integer >= 1 (1-based), got {self.page_number}"
            )
        if self.content is None:
            raise ValueError("content cannot be None")
        if isinstance(self.content, (str, bytes)) and len(self.content) == 0:
            raise ValueError("content cannot be empty")
        if not isinstance(self.related_element_ids, list):
            raise TypeError(
                f"related_element_ids must be a list, got {type(self.related_element_ids).__name__}"
            )
        if self.embedding is not None and not isinstance(
            self.embedding, (list, tuple, np.ndarray)
        ):
            raise TypeError(
                f"embedding must be a list or ndarray, got {type(self.embedding).__name__}"
            )

    @property
    def relationships(self) -> List[str]:
        """Alias for related_element_ids."""
        return self.related_element_ids

    @property
    def related_ids(self) -> List[str]:
        """Alias for related_element_ids."""
        return self.related_element_ids

    @property
    def text(self) -> str:
        """Returns string content or empty string if content is binary."""
        if isinstance(self.content, str):
            return self.content
        return ""

    @property
    def image_data(self) -> bytes:
        """Returns binary image data if content or metadata holds bytes, else b''."""
        if isinstance(self.content, bytes):
            return self.content
        data = self.metadata.get("image_data")
        return data if isinstance(data, bytes) else b""

    @property
    def image_bytes(self) -> bytes:
        """Alias for image_data."""
        return self.image_data

    @property
    def image_path(self) -> Optional[str]:
        """Returns image path if present in metadata or content."""
        if "image_path" in self.metadata and self.metadata["image_path"]:
            return str(self.metadata["image_path"])
        if isinstance(self.content, str) and self.is_image:
            return self.content
        return None

    @property
    def is_text(self) -> bool:
        return self.element_type == ElementType.TEXT

    @property
    def is_image(self) -> bool:
        return self.element_type == ElementType.IMAGE

    @property
    def is_caption(self) -> bool:
        return self.element_type == ElementType.CAPTION

    @property
    def is_table(self) -> bool:
        return self.element_type == ElementType.TABLE

    @property
    def is_equation(self) -> bool:
        return self.element_type == ElementType.EQUATION

    def add_relationship(self, related_id: str) -> None:
        """Adds a related element ID to this element if not already present."""
        if not related_id or not str(related_id).strip():
            raise ValueError("related_id cannot be empty")
        cleaned = str(related_id).strip()
        if cleaned not in self.related_element_ids:
            self.related_element_ids.append(cleaned)

    def add_related_id(self, related_id: str) -> None:
        """Alias for add_relationship."""
        self.add_relationship(related_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the element to a dictionary representation."""
        if isinstance(self.content, bytes):
            content_repr = f"<{len(self.content)} bytes>"
        else:
            content_repr = str(self.content)

        embedding_repr: Optional[List[float]] = None
        if self.embedding is not None:
            embedding_repr = (
                self.embedding.tolist()
                if isinstance(self.embedding, np.ndarray)
                else list(self.embedding)
            )

        return {
            "element_id": self.element_id,
            "element_type": self.element_type.value,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "content": content_repr,
            "embedding": embedding_repr,
            "related_element_ids": list(self.related_element_ids),
            "metadata": {
                k: v for k, v in self.metadata.items() if not isinstance(v, bytes)
            },
        }

    # Classmethod adapters bound to the factory functions below
    @classmethod
    def from_text_chunk(cls, chunk: TextChunk, **kwargs: Any) -> MultimodalElement:
        return from_text_chunk(chunk, **kwargs)

    @classmethod
    def from_image(cls, image: ExtractedImage, **kwargs: Any) -> MultimodalElement:
        return from_image(image, **kwargs)

    @classmethod
    def from_caption(cls, caption: ExtractedCaption, **kwargs: Any) -> MultimodalElement:
        return from_caption(caption, **kwargs)

    @classmethod
    def from_table(cls, table: ExtractedTable, **kwargs: Any) -> MultimodalElement:
        return from_table(table, **kwargs)

    @classmethod
    def from_equation(cls, equation: ExtractedEquation, **kwargs: Any) -> MultimodalElement:
        return from_equation(equation, **kwargs)

    @classmethod
    def from_element(cls, element: Any, **kwargs: Any) -> MultimodalElement:
        return to_multimodal_element(element, **kwargs)


# ---------------------------------------------------------------------------
# Factory / Adapter functions
# ---------------------------------------------------------------------------


def from_text_chunk(
    chunk: TextChunk,
    page_number: Optional[int] = None,
    related_element_ids: Optional[List[str]] = None,
    **kwargs: Any,
) -> MultimodalElement:
    """Adapts a TextChunk into a unified MultimodalElement."""
    if chunk is None:
        raise ValueError("chunk cannot be None")
    if not hasattr(chunk, "chunk_id"):
        raise TypeError("chunk must be a TextChunk instance")

    if page_number is not None:
        resolved_page = int(page_number)
    elif getattr(chunk, "page_numbers", None):
        resolved_page = int(chunk.page_numbers[0])
    else:
        raise ValueError("TextChunk has no page_numbers and page_number was not provided")

    metadata = {
        "chunk_index": getattr(chunk, "chunk_index", 0),
        "page_numbers": list(getattr(chunk, "page_numbers", [resolved_page])),
    }
    if kwargs:
        metadata.update(kwargs)

    return MultimodalElement(
        element_id=chunk.chunk_id,
        element_type=ElementType.TEXT,
        source_filename=chunk.source_filename,
        page_number=resolved_page,
        content=chunk.text,
        related_element_ids=related_element_ids,
        metadata=metadata,
    )


def from_image(
    image: ExtractedImage,
    related_element_ids: Optional[List[str]] = None,
    matches: Optional[Union[Sequence[CaptionMatch], CaptionMatch]] = None,
    **kwargs: Any,
) -> MultimodalElement:
    """Adapts an ExtractedImage into a unified MultimodalElement."""
    if image is None:
        raise ValueError("image cannot be None")
    if not hasattr(image, "image_id"):
        raise TypeError("image must be an ExtractedImage instance")

    if image.image_data:
        content: Union[str, bytes] = image.image_data
    elif image.image_path:
        content = str(image.image_path)
    else:
        content = f"image:{image.image_id}"

    rels = list(related_element_ids or [])
    if matches:
        m_list = [matches] if isinstance(matches, CaptionMatch) else matches
        for m in m_list:
            if getattr(m, "element_id", None) == image.image_id:
                cap_id = getattr(m, "caption_id", None)
                if cap_id and cap_id not in rels:
                    rels.append(cap_id)

    metadata = {
        "image_index": image.image_index,
        "image_format": image.image_format,
        "image_path": image.image_path,
        "image_data": image.image_data,
        "width": image.width,
        "height": image.height,
    }
    if kwargs:
        metadata.update(kwargs)

    return MultimodalElement(
        element_id=image.image_id,
        element_type=ElementType.IMAGE,
        source_filename=image.source_filename,
        page_number=image.page_number,
        content=content,
        related_element_ids=rels,
        metadata=metadata,
    )


def from_caption(
    caption: ExtractedCaption,
    related_element_ids: Optional[List[str]] = None,
    matches: Optional[Union[Sequence[CaptionMatch], CaptionMatch]] = None,
    **kwargs: Any,
) -> MultimodalElement:
    """Adapts an ExtractedCaption into a unified MultimodalElement."""
    if caption is None:
        raise ValueError("caption cannot be None")
    if not hasattr(caption, "caption_id"):
        raise TypeError("caption must be an ExtractedCaption instance")

    rels = list(related_element_ids or [])
    if matches:
        m_list = [matches] if isinstance(matches, CaptionMatch) else matches
        for m in m_list:
            if getattr(m, "caption_id", None) == caption.caption_id:
                elem_id = getattr(m, "element_id", None)
                if elem_id and elem_id not in rels:
                    rels.append(elem_id)

    metadata = {
        "caption_index": caption.caption_index,
        "caption_type": caption.caption_type,
        "label": caption.label,
        "bbox": caption.bbox,
    }
    if kwargs:
        metadata.update(kwargs)

    return MultimodalElement(
        element_id=caption.caption_id,
        element_type=ElementType.CAPTION,
        source_filename=caption.source_filename,
        page_number=caption.page_number,
        content=caption.text,
        related_element_ids=rels,
        metadata=metadata,
    )


def from_table(
    table: ExtractedTable,
    related_element_ids: Optional[List[str]] = None,
    matches: Optional[Union[Sequence[CaptionMatch], CaptionMatch]] = None,
    **kwargs: Any,
) -> MultimodalElement:
    """Adapts an ExtractedTable into a unified MultimodalElement."""
    if table is None:
        raise ValueError("table cannot be None")
    if not hasattr(table, "table_id"):
        raise TypeError("table must be an ExtractedTable instance")

    rels = list(related_element_ids or [])
    if matches:
        m_list = [matches] if isinstance(matches, CaptionMatch) else matches
        for m in m_list:
            if getattr(m, "element_id", None) == table.table_id:
                cap_id = getattr(m, "caption_id", None)
                if cap_id and cap_id not in rels:
                    rels.append(cap_id)

    metadata = {
        "table_index": table.table_index,
        "num_rows": table.num_rows,
        "num_cols": table.num_cols,
        "rows": table.rows,
        "headers": table.headers,
        "bbox": table.bbox,
    }
    if kwargs:
        metadata.update(kwargs)

    return MultimodalElement(
        element_id=table.table_id,
        element_type=ElementType.TABLE,
        source_filename=table.source_filename,
        page_number=table.page_number,
        content=table.content,
        related_element_ids=rels,
        metadata=metadata,
    )


def from_equation(
    equation: ExtractedEquation,
    related_element_ids: Optional[List[str]] = None,
    **kwargs: Any,
) -> MultimodalElement:
    """Adapts an ExtractedEquation into a unified MultimodalElement."""
    if equation is None:
        raise ValueError("equation cannot be None")
    if not hasattr(equation, "equation_id"):
        raise TypeError("equation must be an ExtractedEquation instance")

    metadata = {
        "equation_index": equation.equation_index,
        "equation_label": equation.equation_label,
        "bbox": equation.bbox,
        "is_display": equation.is_display,
    }
    if kwargs:
        metadata.update(kwargs)

    return MultimodalElement(
        element_id=equation.equation_id,
        element_type=ElementType.EQUATION,
        source_filename=equation.source_filename,
        page_number=equation.page_number,
        content=equation.text,
        related_element_ids=related_element_ids,
        metadata=metadata,
    )


def to_multimodal_element(
    element: Any,
    related_element_ids: Optional[List[str]] = None,
    matches: Optional[Union[Sequence[CaptionMatch], CaptionMatch]] = None,
    **kwargs: Any,
) -> MultimodalElement:
    """Universal adapter converting any supported element into a MultimodalElement."""
    if isinstance(element, MultimodalElement):
        return element
    if isinstance(element, TextChunk):
        return from_text_chunk(element, related_element_ids=related_element_ids, **kwargs)
    if isinstance(element, ExtractedImage):
        return from_image(
            element, related_element_ids=related_element_ids, matches=matches, **kwargs
        )
    if isinstance(element, ExtractedCaption):
        return from_caption(
            element, related_element_ids=related_element_ids, matches=matches, **kwargs
        )
    if isinstance(element, ExtractedTable):
        return from_table(
            element, related_element_ids=related_element_ids, matches=matches, **kwargs
        )
    if isinstance(element, ExtractedEquation):
        return from_equation(element, related_element_ids=related_element_ids, **kwargs)
    raise TypeError(f"Cannot convert {type(element).__name__} to MultimodalElement")


def attach_caption_matches(
    elements: Sequence[MultimodalElement],
    matches: Sequence[CaptionMatch],
) -> None:
    """Bidirectionally associates elements with captions based on CaptionMatch instances."""
    by_id = {el.element_id: el for el in elements}
    for match in matches:
        elem = by_id.get(match.element_id)
        cap = by_id.get(match.caption_id)
        if elem is not None:
            elem.add_relationship(match.caption_id)
        if cap is not None:
            cap.add_relationship(match.element_id)
