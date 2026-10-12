from dataclasses import dataclass
from typing import List

from app.chunking.models import TextChunk


@dataclass
class SearchResult:
    """Represents a similarity search result matching a query.

    Attributes:
        chunk: The matched TextChunk object.
        score: Cosine similarity score between the query and chunk.
        chunk_id: Unique identifier of the chunk.
        source_filename: Provenance source filename of the document.
        page_numbers: 1-based page numbers from which the chunk originated.
    """

    chunk: TextChunk
    score: float
    chunk_id: str
    source_filename: str
    page_numbers: List[int]
