"""Sparse BM25 indexing and retrieval for multimodal scientific paper elements.

This module provides in-memory Okapi BM25 sparse retrieval over text-like
MultimodalElement instances (TEXT, CAPTION, TABLE, EQUATION).
IMAGE elements are strictly excluded from the BM25 corpus.
Preserves full original MultimodalElement objects with complete provenance,
returning existing MultimodalSearchResult structures.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from app.indexing.multimodal_models import MultimodalSearchResult
from app.multimodal.representation import ElementType, MultimodalElement

# Deterministic lightweight token regex:
# Captures alphanumeric sequences, preserving hyphenated and underscore-connected
# scientific and identifier terms (e.g., 'd_k', 'w_q', 'bleu-4', 'transformer-base').
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*")


def tokenize(text: str) -> List[str]:
    """Deterministically tokenizes text into normalized lowercase alphanumeric terms.

    Extracts alphanumeric tokens while preserving hyphenated and underscore-connected
    scientific or model terms (such as 'd_k', 'bleu-4', 'gpt-4'). Strips out formatting
    symbols and punctuation from equations and markdown tables.

    Args:
        text: Input string to tokenize.

    Returns:
        List of normalized lowercase token strings.
    """
    if not text or not isinstance(text, str):
        return []
    return _TOKEN_PATTERN.findall(text.lower())


# Alias for backward-compatible or descriptive usage
tokenize_text = tokenize


class BM25Index:
    """In-memory Okapi BM25 index for sparse retrieval over text-like MultimodalElement instances.

    - Indexes only TEXT, CAPTION, TABLE, and EQUATION elements.
    - IMAGE elements are strictly excluded and never enter the BM25 corpus.
    - Preserves full original MultimodalElement objects with provenance and metadata.
    - Implements standard BM25 scoring with configurable k1 and b parameters.
    """

    DEFAULT_K1: float = 1.5
    DEFAULT_B: float = 0.75

    TEXT_LIKE_TYPES: Set[ElementType] = {
        ElementType.TEXT,
        ElementType.CAPTION,
        ElementType.TABLE,
        ElementType.EQUATION,
    }

    def __init__(
        self,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        """Initializes the BM25 index with configurable parameters.

        Args:
            k1: Term frequency saturation parameter (must be >= 0). Defaults to 1.5.
            b: Document length normalization parameter (must be between 0.0 and 1.0). Defaults to 0.75.
        """
        if isinstance(k1, bool) or not isinstance(k1, (int, float)):
            raise TypeError(f"k1 must be a float or int, got {type(k1).__name__}")
        if isinstance(b, bool) or not isinstance(b, (int, float)):
            raise TypeError(f"b must be a float or int, got {type(b).__name__}")

        if k1 < 0.0:
            raise ValueError(f"k1 must be non-negative, got {k1}")
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"b must be between 0.0 and 1.0, got {b}")

        self.k1 = float(k1)
        self.b = float(b)

        self._elements: List[MultimodalElement] = []
        self._doc_tokens: List[List[str]] = []
        self._doc_lengths: List[int] = []
        self._doc_term_freqs: List[Counter[str]] = []
        self._id_to_index: Dict[str, int] = {}
        self._doc_freqs: Dict[str, int] = defaultdict(int)
        self._inverted_index: Dict[str, List[int]] = defaultdict(list)
        self._total_doc_length: int = 0

    @property
    def total_elements(self) -> int:
        """Total number of text-like elements currently indexed."""
        return len(self._elements)

    def __len__(self) -> int:
        return self.total_elements

    @property
    def avgdl(self) -> float:
        """Average document length (in tokens) across all indexed text-like documents."""
        if not self._elements:
            return 0.0
        return self._total_doc_length / len(self._elements)

    @property
    def elements(self) -> List[MultimodalElement]:
        """Returns a copy of all indexed MultimodalElement instances."""
        return list(self._elements)

    def get_element_by_id(self, element_id: str) -> Optional[MultimodalElement]:
        """Retrieves an indexed MultimodalElement by its unique string element_id."""
        idx = self._id_to_index.get(element_id)
        if idx is not None:
            return self._elements[idx]
        return None

    def add_elements(self, elements: Sequence[MultimodalElement]) -> None:
        """Adds a collection/list of MultimodalElement instances to the BM25 index.

        Operates ONLY on text-like elements (TEXT, CAPTION, TABLE, EQUATION).
        IMAGE elements are strictly excluded and never enter the corpus.

        Args:
            elements: Sequence of MultimodalElement instances to index.
        """
        if elements is None:
            raise ValueError("elements cannot be None")

        for idx, el in enumerate(elements):
            if not isinstance(el, MultimodalElement):
                raise TypeError(
                    f"Element at index {idx} must be a MultimodalElement instance, got {type(el).__name__}"
                )

            # Strict exclusion: IMAGE elements never enter BM25 corpus
            if el.element_type == ElementType.IMAGE or el.element_type not in self.TEXT_LIKE_TYPES:
                continue

            if el.element_id in self._id_to_index:
                raise ValueError(
                    f"Duplicate element_id detected: '{el.element_id}' is already present in BM25 index"
                )

            # Extract text content from element
            raw_content = el.content
            text = raw_content if isinstance(raw_content, str) else str(raw_content)

            tokens = tokenize(text)
            term_freq = Counter(tokens)
            doc_len = len(tokens)

            doc_id = len(self._elements)
            self._elements.append(el)
            self._doc_tokens.append(tokens)
            self._doc_lengths.append(doc_len)
            self._doc_term_freqs.append(term_freq)
            self._id_to_index[el.element_id] = doc_id
            self._total_doc_length += doc_len

            # Update document frequencies and inverted index
            for term in term_freq.keys():
                self._doc_freqs[term] += 1
                self._inverted_index[term].append(doc_id)

    def add_document(self, document: Any) -> None:
        """Adds all text-like elements from a MultimodalDocument to the BM25 index.

        Any IMAGE elements present in the document are strictly excluded.

        Args:
            document: MultimodalDocument instance (or any object with an 'elements' attribute).
        """
        if document is None:
            raise ValueError("document cannot be None")
        if not hasattr(document, "elements"):
            raise TypeError(
                f"document must have an 'elements' attribute, got {type(document).__name__}"
            )
        self.add_elements(document.elements)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[MultimodalSearchResult]:
        """Performs BM25 similarity search matching a text query against the indexed corpus.

        Args:
            query: Query text string.
            top_k: Maximum number of top ranked results to return (must be positive).

        Returns:
            List of MultimodalSearchResult instances ranked by BM25 score descending.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if self.total_elements == 0:
            return []

        if query is None:
            return []

        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")

        if not query.strip():
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        num_docs = len(self._elements)
        avg_doc_len = self.avgdl
        scores: Dict[int, float] = defaultdict(float)

        for q_term in query_tokens:
            if q_term not in self._doc_freqs:
                continue

            doc_freq = self._doc_freqs[q_term]
            # Standard Robertson-Spärck Jones / Lucene IDF formulation (strictly non-negative)
            idf = math.log(1.0 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))

            for doc_id in self._inverted_index[q_term]:
                tf = self._doc_term_freqs[doc_id][q_term]
                doc_len = self._doc_lengths[doc_id]

                # Document length normalization factor
                if avg_doc_len > 0.0:
                    len_norm = 1.0 - self.b + self.b * (doc_len / avg_doc_len)
                else:
                    len_norm = 1.0

                tf_component = (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
                scores[doc_id] += idf * tf_component

        # Filter out zero or negative scores
        candidate_scores: List[Tuple[float, int]] = [
            (score, doc_id) for doc_id, score in scores.items() if score > 0.0
        ]

        if not candidate_scores:
            return []

        # Sort descending by score, breaking ties deterministically by doc_id
        candidate_scores.sort(key=lambda item: (-item[0], item[1]))

        # Limit to top_k
        top_candidates = candidate_scores[:top_k]

        results: List[MultimodalSearchResult] = []
        for score, doc_id in top_candidates:
            element = self._elements[doc_id]
            results.append(
                MultimodalSearchResult(
                    element=element,
                    score=float(score),
                    element_id=element.element_id,
                    element_type=element.element_type,
                    source_filename=element.source_filename,
                    page_number=element.page_number,
                    related_element_ids=list(element.related_element_ids),
                )
            )

        return results

    def clear(self) -> None:
        """Clears all indexed elements and resets the index state."""
        self._elements.clear()
        self._doc_tokens.clear()
        self._doc_lengths.clear()
        self._doc_term_freqs.clear()
        self._id_to_index.clear()
        self._doc_freqs.clear()
        self._inverted_index.clear()
        self._total_doc_length = 0

    @classmethod
    def from_elements(
        cls,
        elements: Sequence[MultimodalElement],
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> "BM25Index":
        """Builds a new BM25Index populated with the provided multimodal elements."""
        index = cls(k1=k1, b=b)
        index.add_elements(elements)
        return index

    @classmethod
    def from_document(
        cls,
        document: Any,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> "BM25Index":
        """Builds a new BM25Index directly from a MultimodalDocument."""
        index = cls(k1=k1, b=b)
        index.add_document(document)
        return index


def build_bm25_index(
    source: Union[Any, Sequence[MultimodalElement]],
    k1: float = BM25Index.DEFAULT_K1,
    b: float = BM25Index.DEFAULT_B,
) -> BM25Index:
    """Convenience factory function to build a BM25Index from a document or element list."""
    if hasattr(source, "elements"):
        return BM25Index.from_document(document=source, k1=k1, b=b)
    if isinstance(source, (list, tuple)):
        return BM25Index.from_elements(elements=source, k1=k1, b=b)
    raise TypeError(
        f"source must be a MultimodalDocument or Sequence[MultimodalElement], got {type(source).__name__}"
    )
