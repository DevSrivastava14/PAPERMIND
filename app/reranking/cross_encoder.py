"""Cross-encoder based candidate reranker for PaperMind.

Provides high-precision neural reranking of candidate MultimodalSearchResult items
using Sentence Transformers CrossEncoder models while preserving all original
provenance, multimodal element references, and metadata.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from app.indexing.multimodal_models import MultimodalSearchResult
from app.reranking.base import BaseReranker

DEFAULT_CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BATCH_SIZE: int = 32


class ScoreCountMismatchError(RuntimeError, ValueError):
    """Raised when the number of scores returned by the model does not match the candidate count."""

    pass


def default_candidate_text_extractor(candidate: MultimodalSearchResult) -> str:
    """Extracts a textual representation from a candidate MultimodalSearchResult.

    Handles text-like elements directly (TEXT, CAPTION, TABLE, EQUATION), and
    safely handles IMAGE elements via metadata (caption, text, description)
    without exposing binary payloads.
    """
    element = candidate.element
    if element.text:
        return element.text.strip()
    if isinstance(element.content, str):
        return element.content.strip()
    if isinstance(element.metadata, dict):
        for key in ("caption", "text", "description", "title", "label"):
            val = element.metadata.get(key)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
    return ""


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder based candidate reranker for MultimodalSearchResult candidates.

    Evaluates (query, candidate_text) pairs jointly using a cross-encoder model,
    returning candidates ordered by descending relevance score with deterministic
    tie-breaking while strictly preserving all original MultimodalSearchResult
    provenance, element references, and metadata fields.
    """

    DEFAULT_MODEL_NAME: str = DEFAULT_CROSS_ENCODER_MODEL
    DEFAULT_BATCH_SIZE: int = DEFAULT_BATCH_SIZE

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model: Optional[Any] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        text_extractor: Optional[Callable[[MultimodalSearchResult], str]] = None,
        **model_kwargs: Any,
    ) -> None:
        """Initializes the cross-encoder reranker with configurable model and parameters.

        Args:
            model_name: Sentence Transformers model identifier.
            model: Optional pre-loaded or mock model instance. If provided, model
                weights are not loaded from disk or downloaded.
            batch_size: Batch size for cross-encoder inference (must be positive).
            device: Compute device string (e.g., 'cpu', 'cuda').
            max_length: Optional maximum sequence length for the tokenizer.
            text_extractor: Optional custom function extracting text from a candidate.
            **model_kwargs: Additional keyword arguments passed to CrossEncoder on load.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError(f"batch_size must be an integer, got {type(batch_size).__name__}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if max_length is not None:
            if isinstance(max_length, bool) or not isinstance(max_length, int):
                raise TypeError(f"max_length must be an integer, got {type(max_length).__name__}")
            if max_length <= 0:
                raise ValueError(f"max_length must be positive, got {max_length}")

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        self.model_name = model_name.strip()
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self._text_extractor = text_extractor or default_candidate_text_extractor
        self._model_kwargs = dict(model_kwargs)

        self._model = model

    @property
    def model(self) -> Any:
        """Returns the underlying cross-encoder model instance, lazy-loading if needed."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            init_kwargs = dict(self._model_kwargs)
            if self.device is not None:
                init_kwargs["device"] = self.device
            if self.max_length is not None:
                init_kwargs["max_length"] = self.max_length

            self._model = CrossEncoder(self.model_name, **init_kwargs)
        return self._model

    @staticmethod
    def _normalize_scores(raw_scores: Any, expected_count: int) -> List[float]:
        """Normalizes model outputs into a flat list of float scores with count validation."""
        if raw_scores is None:
            raise ScoreCountMismatchError(
                f"Cross-encoder returned None, expected {expected_count} scores"
            )

        # Handle NumPy arrays or array-like objects
        if hasattr(raw_scores, "shape"):
            import numpy as np

            arr = np.asarray(raw_scores)
            if arr.ndim == 0 and expected_count == 1:
                return [float(arr.item())]
            elif arr.ndim == 2:
                if arr.shape[1] == 1:
                    arr = arr.squeeze(axis=1)
                elif arr.shape[1] >= 2:
                    # Binary classification logits: use positive class score
                    arr = arr[:, 1]
                else:
                    arr = arr.flatten()
            elif arr.ndim > 2:
                arr = arr.flatten()
            scores = [float(x) for x in arr]
        elif isinstance(raw_scores, (list, tuple)):
            scores = []
            for item in raw_scores:
                if isinstance(item, (list, tuple)):
                    if len(item) == 1:
                        scores.append(float(item[0]))
                    elif len(item) >= 2:
                        scores.append(float(item[1]))
                    else:
                        scores.append(0.0)
                elif hasattr(item, "item"):
                    scores.append(float(item.item()))
                else:
                    scores.append(float(item))
        else:
            try:
                scores = [float(raw_scores)]
            except (TypeError, ValueError):
                raise ScoreCountMismatchError(
                    f"Unexpected score output format: {type(raw_scores).__name__}"
                )

        if len(scores) != expected_count:
            raise ScoreCountMismatchError(
                f"Cross-encoder returned {len(scores)} scores for {expected_count} candidates"
            )

        return scores

    def rerank(
        self,
        query: str,
        candidates: Sequence[MultimodalSearchResult],
        top_k: Optional[int] = None,
    ) -> List[MultimodalSearchResult]:
        """Reranks candidate search results against a query string.

        Args:
            query: Non-empty query string.
            candidates: Sequence of MultimodalSearchResult candidates to rerank.
            top_k: Optional positive integer limiting the maximum number of returned results.

        Returns:
            List of MultimodalSearchResult instances sorted by reranked score descending,
            with deterministic tie-breaking by element_id.

        Raises:
            TypeError: If query, top_k, candidates, or candidate items have invalid types.
            ValueError: If query is empty/whitespace or top_k is non-positive.
            ScoreCountMismatchError: If the model returns an unexpected number of scores.
        """
        # Validate query
        if query is None or not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("query cannot be empty or whitespace-only")

        # Validate top_k
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int):
                raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}")
            if top_k <= 0:
                raise ValueError(f"top_k must be a positive integer, got {top_k}")

        # Validate candidates container
        if candidates is None:
            raise TypeError("candidates cannot be None")
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, (list, tuple, Sequence)):
            raise TypeError(f"candidates must be a sequence, got {type(candidates).__name__}")

        # Validate candidate items
        for idx, item in enumerate(candidates):
            if not isinstance(item, MultimodalSearchResult):
                raise TypeError(
                    f"Candidate at index {idx} must be a MultimodalSearchResult instance, got {type(item).__name__}"
                )

        if not candidates:
            return []

        # Build (query, text) pairs
        cleaned_query = query.strip()
        pairs = [[cleaned_query, self._text_extractor(c)] for c in candidates]

        # Invoke model
        mdl = self.model
        try:
            raw_scores = mdl.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except TypeError:
            raw_scores = mdl.predict(pairs)

        scores = self._normalize_scores(raw_scores, expected_count=len(candidates))

        # Pair candidates with their respective scores
        scored_pairs = list(zip(candidates, scores))

        # Deterministic sorting: descending by score, tie-break by element_id ascending
        sorted_scored = sorted(
            scored_pairs,
            key=lambda pair: (-pair[1], str(pair[0].element_id or "")),
        )

        if top_k is not None:
            sorted_scored = sorted_scored[:top_k]

        # Reconstruct MultimodalSearchResult preserving every provenance and element field
        reranked_results: List[MultimodalSearchResult] = []
        for orig, new_score in sorted_scored:
            reranked_results.append(
                MultimodalSearchResult(
                    element=orig.element,
                    score=float(new_score),
                    element_id=orig.element_id,
                    element_type=orig.element_type,
                    source_filename=orig.source_filename,
                    page_number=orig.page_number,
                    related_element_ids=list(orig.related_element_ids or []),
                )
            )

        return reranked_results


def rerank_cross_encoder(
    query: str,
    candidates: Sequence[MultimodalSearchResult],
    reranker: Optional[CrossEncoderReranker] = None,
    top_k: Optional[int] = None,
    model: Optional[Any] = None,
) -> List[MultimodalSearchResult]:
    """Convenience function to rerank candidates using a CrossEncoderReranker.

    Args:
        query: Search query text.
        candidates: Sequence of MultimodalSearchResult candidates.
        reranker: Optional existing CrossEncoderReranker instance.
        top_k: Optional limit on the number of returned results.
        model: Optional pre-loaded or mock model (used if reranker is not provided).

    Returns:
        List of reranked MultimodalSearchResult instances.
    """
    if reranker is None:
        reranker = CrossEncoderReranker(model=model)
    elif model is not None:
        reranker = CrossEncoderReranker(model=model)

    return reranker.rerank(query=query, candidates=candidates, top_k=top_k)
