from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    """Represents a text chunk extracted from a research paper document."""

    chunk_id: str
    text: str
    source_filename: str
    chunk_index: int
    page_numbers: List[int]
