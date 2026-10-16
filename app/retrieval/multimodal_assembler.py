"""Context assembly layer for multimodal search results.

This module formats heterogeneous MultimodalSearchResult objects (text, tables, equations,
captions, and images) into prompt-ready evidence blocks with sequential citation labels,
provenance tracking, and cross-element relationship (e.g. image <-> caption) resolution.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

from app.indexing.multimodal_models import MultimodalSearchResult
from app.multimodal.collection import MultimodalDocument
from app.multimodal.representation import ElementType, MultimodalElement
from app.retrieval.multimodal_models import (
    MultimodalCitation,
    MultimodalRetrievedContext,
)


class MultimodalContextAssembler:
    """Assembles retrieved MultimodalSearchResult items into prompt-ready context and citations."""

    def __init__(self, document: Optional[MultimodalDocument] = None) -> None:
        self.default_document = document

    def _format_element_body(
        self,
        result: MultimodalSearchResult,
        doc: Optional[MultimodalDocument],
    ) -> str:
        """Formats the element content based on its modality type.

        Ensures raw image bytes are never emitted into prompt text.
        """
        element = result.element
        el_type = result.element_type

        if el_type == ElementType.TEXT:
            text_val = element.text.strip() if element.text else str(element.content).strip()
            return text_val

        elif el_type == ElementType.CAPTION:
            caption_val = element.text.strip() if element.text else str(element.content).strip()
            return caption_val

        elif el_type == ElementType.TABLE:
            table_val = element.text.strip() if element.text else str(element.content).strip()
            return table_val

        elif el_type == ElementType.EQUATION:
            eq_val = element.text.strip() if element.text else str(element.content).strip()
            return eq_val

        elif el_type == ElementType.IMAGE:
            # Build image descriptor without raw bytes
            descriptor_lines = [f"[Image: {result.element_id}]"]

            # Format/dimension metadata
            meta = result.metadata or {}
            meta_details = []
            if "image_format" in meta and meta["image_format"]:
                meta_details.append(f"format: {meta['image_format']}")
            if "width" in meta and "height" in meta:
                meta_details.append(f"dimensions: {meta['width']}x{meta['height']}")
            if element.image_path:
                meta_details.append(f"path: {element.image_path}")

            if meta_details:
                descriptor_lines.append(f"Image Details: {', '.join(meta_details)}")

            # Relationship handling: resolve linked caption if document is available
            resolved_caption: Optional[str] = None
            if doc is not None and result.related_element_ids:
                for rel_id in result.related_element_ids:
                    rel_el = doc.get_by_id(rel_id)
                    if rel_el is not None and rel_el.text:
                        resolved_caption = rel_el.text.strip()
                        break

            if resolved_caption:
                descriptor_lines.append(f"Caption: {resolved_caption}")
            elif result.related_element_ids:
                descriptor_lines.append(
                    f"Related Element IDs: {', '.join(result.related_element_ids)}"
                )

            return "\n".join(descriptor_lines)

        else:
            # Fallback for unknown element types
            fallback = element.text.strip() if element.text else str(element.content).strip()
            return fallback

    def assemble(
        self,
        search_results: Sequence[MultimodalSearchResult],
        document: Optional[MultimodalDocument] = None,
    ) -> MultimodalRetrievedContext:
        """Formats search results into delimited evidence blocks with sequential citations.

        Args:
            search_results: Ordered sequence of MultimodalSearchResult items.
            document: Optional MultimodalDocument used to resolve inter-element relationships.

        Returns:
            MultimodalRetrievedContext with formatted text, structured citations, and results.
        """
        if not search_results:
            return MultimodalRetrievedContext(
                formatted_context="",
                citations=[],
                search_results=[],
            )

        doc = document if document is not None else self.default_document
        citations: List[MultimodalCitation] = []
        evidence_blocks: List[str] = []

        for idx, res in enumerate(search_results, start=1):
            label = f"[{idx}]"
            type_label = (
                res.element_type.name.capitalize()
                if isinstance(res.element_type, ElementType)
                else str(res.element_type).upper()
            )
            page_ref = f"Page {res.page_number}"

            citation = MultimodalCitation(
                citation_label=label,
                element_id=res.element_id,
                element_type=res.element_type,
                source_filename=res.source_filename,
                page_number=res.page_number,
                score=res.score,
            )
            citations.append(citation)

            content_body = self._format_element_body(res, doc)
            header = f"{label} [{type_label}] Source: {res.source_filename} ({page_ref})"
            block = f"{header}\n{content_body}"
            evidence_blocks.append(block)

        formatted_context = "\n\n".join(evidence_blocks)

        return MultimodalRetrievedContext(
            formatted_context=formatted_context,
            citations=citations,
            search_results=list(search_results),
        )


def assemble_multimodal_context(
    search_results: Sequence[MultimodalSearchResult],
    document: Optional[MultimodalDocument] = None,
) -> MultimodalRetrievedContext:
    """Convenience function to assemble MultimodalSearchResult items into a MultimodalRetrievedContext."""
    assembler = MultimodalContextAssembler(document=document)
    return assembler.assemble(search_results, document=document)
