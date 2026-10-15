"""Multimodal document and collection abstraction for scientific papers.

This module provides the collection and assembly layer (`MultimodalDocument` /
`MultimodalCollection`) that aggregates unified multimodal elements extracted
from a paper, maintains relationship graphs (e.g. image <-> caption, table <-> caption),
and integrates with `MultimodalEmbedder` to generate and query modality-specific embeddings.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np

from app.chunking.models import TextChunk
from app.multimodal.embedding import (
    BaseMultimodalEmbedder,
    MultimodalEmbeddingResult,
    embed_multimodal_elements,
)
from app.multimodal.models import (
    CaptionMatch,
    ExtractedCaption,
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
)
from app.multimodal.representation import (
    ElementType,
    MultimodalElement,
    attach_caption_matches,
    from_caption,
    from_equation,
    from_image,
    from_table,
    from_text_chunk,
)


class MultimodalDocument:
    """A coherent collection of multimodal elements representing a scientific paper.

    Provides indexed lookups by element ID, element type, page number, relationship
    traversals, and embedding integration while preserving exact provenance and
    separate modality-appropriate embedding spaces.
    """

    def __init__(
        self,
        elements: Optional[Sequence[MultimodalElement]] = None,
        source_filename: Optional[str] = None,
    ) -> None:
        self.elements: List[MultimodalElement] = []
        self._by_id: Dict[str, MultimodalElement] = {}
        self.source_filename = source_filename

        if elements:
            for el in elements:
                self.add_element(el)

        if not self.source_filename and self.elements:
            self.source_filename = self.elements[0].source_filename

    def add_element(self, element: MultimodalElement) -> None:
        """Adds a MultimodalElement to the collection, ensuring unique IDs."""
        if not isinstance(element, MultimodalElement):
            raise TypeError(
                f"element must be a MultimodalElement, got {type(element).__name__}"
            )
        if element.element_id in self._by_id:
            raise ValueError(
                f"Duplicate element_id detected: '{element.element_id}' is already present in document"
            )
        self.elements.append(element)
        self._by_id[element.element_id] = element

        if not self.source_filename:
            self.source_filename = element.source_filename

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self) -> Iterator[MultimodalElement]:
        return iter(self.elements)

    def __getitem__(self, key: Union[int, str]) -> MultimodalElement:
        """Supports indexing by position integer or lookup by element_id string."""
        if isinstance(key, int):
            return self.elements[key]
        if isinstance(key, str):
            el = self.get_by_id(key)
            if el is None:
                raise KeyError(f"Element with ID '{key}' not found in document")
            return el
        raise TypeError(f"Key must be int or str, got {type(key).__name__}")

    def __contains__(self, element_id: str) -> bool:
        return element_id in self._by_id

    @property
    def element_ids(self) -> List[str]:
        """List of all element IDs in original document order."""
        return [el.element_id for el in self.elements]

    @property
    def element_types(self) -> List[ElementType]:
        """List of all element types in original document order."""
        return [el.element_type for el in self.elements]

    def get_by_id(self, element_id: str) -> Optional[MultimodalElement]:
        """Retrieves an element by its unique identifier, or None if not found."""
        return self._by_id.get(element_id)

    def get_by_type(
        self, element_type: Union[ElementType, str]
    ) -> List[MultimodalElement]:
        """Returns all elements matching the specified element type."""
        target = (
            ElementType(element_type)
            if isinstance(element_type, str)
            else element_type
        )
        return [el for el in self.elements if el.element_type == target]

    def get_by_page(self, page_number: int) -> List[MultimodalElement]:
        """Returns all elements appearing on the specified 1-based page number."""
        if not isinstance(page_number, int) or page_number < 1:
            raise ValueError(
                f"page_number must be an integer >= 1, got {page_number}"
            )
        return [el for el in self.elements if el.page_number == page_number]

    def get_related(self, element_id: str) -> List[MultimodalElement]:
        """Resolves and returns all MultimodalElement instances related to element_id.

        E.g. for an image, returns its matched caption element; for a caption,
        returns its matched visual element(s).
        """
        el = self.get_by_id(element_id)
        if el is None:
            return []
        related_elements: List[MultimodalElement] = []
        for rel_id in el.related_element_ids:
            rel_el = self.get_by_id(rel_id)
            if rel_el is not None and rel_el not in related_elements:
                related_elements.append(rel_el)
        return related_elements

    def get_embedding(self, element_id: str) -> Optional[np.ndarray]:
        """Returns the embedding vector for the given element_id, or None if not embedded."""
        el = self.get_by_id(element_id)
        if el is None:
            return None
        return el.embedding

    def embedded_elements(self) -> List[MultimodalElement]:
        """Returns all elements that currently possess an embedding vector."""
        return [el for el in self.elements if el.embedding is not None]

    @property
    def is_fully_embedded(self) -> bool:
        """True if all elements in the collection have been embedded."""
        return len(self.elements) > 0 and len(self.embedded_elements()) == len(
            self.elements
        )

    def embed(
        self,
        embedder: Optional[BaseMultimodalEmbedder] = None,
        batch_size: Optional[int] = None,
    ) -> MultimodalEmbeddingResult:
        """Computes and associates modality-specific embeddings for all elements in the collection.

        Text-like elements receive embeddings from the semantic text embedder (e.g. 384-d),
        while images receive embeddings from the visual embedder (e.g. 512-d).
        Embedding spaces remain separate and are not concatenated.
        """
        result = embed_multimodal_elements(
            self.elements, embedder=embedder, batch_size=batch_size
        )
        return result

    def get_vectors_by_type(
        self, element_type: Union[ElementType, str]
    ) -> np.ndarray:
        """Returns a 2D float32 NumPy array of vectors for elements of the specified type."""
        elements_of_type = self.get_by_type(element_type)
        vectors = [el.embedding for el in elements_of_type if el.embedding is not None]
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the document collection structure to a dictionary."""
        return {
            "source_filename": self.source_filename,
            "element_count": len(self.elements),
            "element_ids": self.element_ids,
            "elements": [el.to_dict() for el in self.elements],
        }

    @classmethod
    def assemble(cls, **kwargs: Any) -> MultimodalDocument:
        """Classmethod helper wrapping assemble_multimodal_document."""
        return assemble_multimodal_document(**kwargs)


MultimodalCollection = MultimodalDocument


# ---------------------------------------------------------------------------
# Assembly Function
# ---------------------------------------------------------------------------


def assemble_multimodal_document(
    text_chunks: Optional[Sequence[TextChunk]] = None,
    images: Optional[Sequence[ExtractedImage]] = None,
    captions: Optional[Sequence[ExtractedCaption]] = None,
    tables: Optional[Sequence[ExtractedTable]] = None,
    equations: Optional[Sequence[ExtractedEquation]] = None,
    caption_matches: Optional[Sequence[CaptionMatch]] = None,
    source_filename: Optional[str] = None,
) -> MultimodalDocument:
    """Assembles heterogeneous extracted scientific paper elements into a unified MultimodalDocument.

    Preserves exact element IDs, 1-based page numbers, original content, modality-specific
    metadata, and bidirectional relationships (image <-> caption, table <-> caption).
    """
    elements: List[MultimodalElement] = []

    # 1. Text chunks
    if text_chunks:
        for chunk in text_chunks:
            elements.append(from_text_chunk(chunk))

    # 2. Images
    if images:
        for img in images:
            elements.append(from_image(img, matches=caption_matches))

    # 3. Captions
    if captions:
        for cap in captions:
            elements.append(from_caption(cap, matches=caption_matches))

    # 4. Tables
    if tables:
        for tbl in tables:
            elements.append(from_table(tbl, matches=caption_matches))

    # 5. Equations
    if equations:
        for eq in equations:
            elements.append(from_equation(eq))

    # 6. Apply bidirectional CaptionMatch relationships across all elements
    if caption_matches:
        attach_caption_matches(elements, caption_matches)

    return MultimodalDocument(
        elements=elements,
        source_filename=source_filename,
    )


assemble_multimodal_collection = assemble_multimodal_document
