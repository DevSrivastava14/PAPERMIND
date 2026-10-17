"""Reciprocal Rank Fusion (RRF) for multimodal retrieval result lists.

This module provides a standalone, independent Reciprocal Rank Fusion component
to combine multiple ranked retrieval result lists (e.g. dense FAISS and sparse BM25)
using the standard reciprocal rank formula:
    RRF_score(d) = sum_{r in R} (1 / (k + rank(d, r)))
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from app.indexing.multimodal_models import MultimodalSearchResult

# Standard default smoothing constant from Cormack et al. (SIGIR 2009)
DEFAULT_RRF_K: int = 60


class ReciprocalRankFusion:
    """Combines multiple ranked lists of MultimodalSearchResult using Reciprocal Rank Fusion.

    Attributes:
        k: Smoothing constant added to rank in denominator (must be positive).
    """

    def __init__(self, k: int = DEFAULT_RRF_K) -> None:
        """Initializes the RRF combiner with configurable k parameter.

        Args:
            k: Smoothing parameter (must be a positive number). Defaults to 60.

        Raises:
            TypeError: If k is not an integer or float.
            ValueError: If k <= 0.
        """
        if isinstance(k, bool) or not isinstance(k, (int, float)):
            raise TypeError(f"k must be an integer or float, got {type(k).__name__}")
        if k <= 0:
            raise ValueError(f"k must be a positive number, got {k}")

        self.k = int(k) if isinstance(k, int) else float(k)

    def fuse(
        self,
        ranked_lists: Sequence[Sequence[MultimodalSearchResult]],
        top_k: Optional[int] = None,
    ) -> List[MultimodalSearchResult]:
        """Fuses multiple ranked result lists into a single deduplicated ranked list.

        Args:
            ranked_lists: Sequence of ranked result lists (each containing MultimodalSearchResult).
            top_k: Optional positive integer limiting the maximum number of returned results.

        Returns:
            List of MultimodalSearchResult with scores replaced by their combined RRF scores,
            ordered by descending RRF score, breaking ties deterministically by element_id.

        Raises:
            TypeError: If ranked_lists or top_k have invalid types, or list items are invalid.
            ValueError: If top_k <= 0.
        """
        if ranked_lists is None:
            raise TypeError("ranked_lists cannot be None")
        if not isinstance(ranked_lists, (list, tuple)):
            raise TypeError(
                f"ranked_lists must be a sequence of sequences, got {type(ranked_lists).__name__}"
            )

        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int):
                raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}")
            if top_k <= 0:
                raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if not ranked_lists:
            return []

        # Validate structure and items
        for list_idx, r_list in enumerate(ranked_lists):
            if r_list is None:
                raise TypeError(f"ranked_lists item at index {list_idx} cannot be None")
            if not isinstance(r_list, (list, tuple)):
                raise TypeError(
                    f"ranked_lists item at index {list_idx} must be a sequence, got {type(r_list).__name__}"
                )
            for item_idx, item in enumerate(r_list):
                if not isinstance(item, MultimodalSearchResult):
                    raise TypeError(
                        f"Item at list {list_idx}, position {item_idx} must be a MultimodalSearchResult, "
                        f"got {type(item).__name__}"
                    )

        # Accumulate RRF scores and record first encountered result object for provenance
        rrf_scores: Dict[str, float] = {}
        provenance_by_id: Dict[str, MultimodalSearchResult] = {}

        for r_list in ranked_lists:
            # Deduplicate within the same ranked list (keep best/first rank)
            seen_in_list = set()
            for rank_0, result in enumerate(r_list):
                elem_id = result.element_id or result.element.element_id
                if elem_id in seen_in_list:
                    continue
                seen_in_list.add(elem_id)

                # 1-based rank
                rank_1 = rank_0 + 1
                contribution = 1.0 / (self.k + rank_1)

                rrf_scores[elem_id] = rrf_scores.get(elem_id, 0.0) + contribution

                if elem_id not in provenance_by_id:
                    provenance_by_id[elem_id] = result

        if not rrf_scores:
            return []

        # Sort descending by RRF score, breaking ties deterministically by element_id
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda eid: (-rrf_scores[eid], eid),
        )

        if top_k is not None:
            sorted_ids = sorted_ids[:top_k]

        fused_results: List[MultimodalSearchResult] = []
        for eid in sorted_ids:
            orig = provenance_by_id[eid]
            fused_results.append(
                MultimodalSearchResult(
                    element=orig.element,
                    score=float(rrf_scores[eid]),
                    element_id=orig.element_id,
                    element_type=orig.element_type,
                    source_filename=orig.source_filename,
                    page_number=orig.page_number,
                    related_element_ids=list(orig.related_element_ids or []),
                )
            )

        return fused_results


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[MultimodalSearchResult]],
    k: int = DEFAULT_RRF_K,
    top_k: Optional[int] = None,
) -> List[MultimodalSearchResult]:
    """Convenience function executing Reciprocal Rank Fusion over multiple ranked result lists.

    Args:
        ranked_lists: Sequence of ranked result lists.
        k: Smoothing parameter (default 60).
        top_k: Optional positive integer limiting the maximum number of returned results.

    Returns:
        List of fused MultimodalSearchResult ranked by RRF score descending.
    """
    fuser = ReciprocalRankFusion(k=k)
    return fuser.fuse(ranked_lists=ranked_lists, top_k=top_k)
