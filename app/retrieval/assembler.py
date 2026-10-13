from typing import List

from app.indexing.models import SearchResult
from app.retrieval.models import Citation, RetrievedContext


def format_page_reference(page_numbers: List[int]) -> str:
    """Formats 1-based page numbers into a clean, human-readable reference string.

    Preserves exact original page numbers without renumbering or inventing pages.
    Examples:
        [1] -> "Page 1"
        [1, 2] -> "Pages 1-2"
        [1, 3] -> "Pages 1, 3"
    """
    if not page_numbers:
        return "Page unknown"

    sorted_pages = sorted(page_numbers)
    if len(sorted_pages) == 1:
        return f"Page {sorted_pages[0]}"

    # Check for contiguous range
    if sorted_pages == list(range(sorted_pages[0], sorted_pages[-1] + 1)):
        return f"Pages {sorted_pages[0]}-{sorted_pages[-1]}"

    return f"Pages {', '.join(str(p) for p in sorted_pages)}"


class CitationContextAssembler:
    """Assembles retrieved SearchResult objects into prompt-ready context and citations."""

    def assemble(self, search_results: List[SearchResult]) -> RetrievedContext:
        """Formats search results into delimited evidence blocks with citations.

        Handles empty results cleanly by returning empty formatted context and citations.
        """
        if not search_results:
            return RetrievedContext(
                formatted_context="",
                citations=[],
                search_results=[],
            )

        citations: List[Citation] = []
        evidence_blocks: List[str] = []

        for idx, res in enumerate(search_results, start=1):
            label = f"[{idx}]"
            page_ref = format_page_reference(res.page_numbers)

            citation = Citation(
                citation_label=label,
                source_filename=res.source_filename,
                page_numbers=list(res.page_numbers),
                chunk_id=res.chunk_id,
                score=res.score,
            )
            citations.append(citation)

            chunk_text = res.chunk.text.strip() if res.chunk and res.chunk.text else ""
            block = f"{label} Source: {res.source_filename} ({page_ref})\n{chunk_text}"
            evidence_blocks.append(block)

        formatted_context = "\n\n".join(evidence_blocks)

        return RetrievedContext(
            formatted_context=formatted_context,
            citations=citations,
            search_results=list(search_results),
        )


def assemble_context(search_results: List[SearchResult]) -> RetrievedContext:
    """Convenience function to assemble SearchResult items into a RetrievedContext."""
    assembler = CitationContextAssembler()
    return assembler.assemble(search_results)
