from dataclasses import dataclass
from typing import List

from app.chunking.models import TextChunk
from app.ingestion.models import ResearchPaperDocument


@dataclass
class _PageSpan:
    page_number: int
    start: int
    end: int


class TextChunker:
    """Chunks research paper documents into deterministic character-based text chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        page_separator: str = "\n\n",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.page_separator = page_separator

    def chunk_document(self, document: ResearchPaperDocument) -> List[TextChunk]:
        """Splits a ResearchPaperDocument into a list of TextChunk instances.

        Preserves accurate page provenance across page boundaries and skips blank pages.
        """
        valid_pages = [
            page
            for page in document.pages
            if page.has_text and page.text and page.text.strip()
        ]
        if not valid_pages:
            return []

        text_parts: List[str] = []
        page_spans: List[_PageSpan] = []
        current_offset = 0

        for idx, page in enumerate(valid_pages):
            page_text = page.text
            start = current_offset
            end = current_offset + len(page_text)
            page_spans.append(_PageSpan(page_number=page.page_number, start=start, end=end))
            text_parts.append(page_text)
            current_offset = end

            if idx < len(valid_pages) - 1:
                text_parts.append(self.page_separator)
                current_offset += len(self.page_separator)

        full_text = "".join(text_parts)
        total_len = len(full_text)
        step_size = self.chunk_size - self.chunk_overlap

        chunks: List[TextChunk] = []
        start_idx = 0
        chunk_idx = 0

        while start_idx < total_len:
            end_idx = min(start_idx + self.chunk_size, total_len)
            chunk_text = full_text[start_idx:end_idx]

            contributing_pages = [
                span.page_number
                for span in page_spans
                if max(start_idx, span.start) < min(end_idx, span.end)
            ]

            if contributing_pages:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document.filename}#chunk_{chunk_idx}",
                        text=chunk_text,
                        source_filename=document.filename,
                        chunk_index=chunk_idx,
                        page_numbers=contributing_pages,
                    )
                )
                chunk_idx += 1

            if end_idx >= total_len:
                break
            start_idx += step_size

        return chunks


def chunk_document(
    document: ResearchPaperDocument,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    page_separator: str = "\n\n",
) -> List[TextChunk]:
    """Convenience function to chunk a document with given parameters."""
    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        page_separator=page_separator,
    )
    return chunker.chunk_document(document)
