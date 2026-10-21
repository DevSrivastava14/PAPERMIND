"""Opt-in multimodal reranking for heterogeneous retrieval candidates."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from app.indexing.multimodal_models import MultimodalSearchResult
from app.multimodal.embedding import BaseVisualEmbedder
from app.multimodal.representation import ElementType
from app.reranking.base import BaseReranker
from app.reranking.cross_encoder import (
    CrossEncoderReranker,
    default_candidate_text_extractor,
)


class MultimodalReranker(BaseReranker):
    """Combines text cross-encoder and shared-space visual scores.

    This is opt-in. Text-only pipelines should continue using CrossEncoderReranker.
    """

    TEXT_LIKE_TYPES = {
        ElementType.TEXT,
        ElementType.CAPTION,
        ElementType.TABLE,
        ElementType.EQUATION,
    }

    def __init__(
        self,
        text_reranker: CrossEncoderReranker,
        visual_embedder: BaseVisualEmbedder,
        text_weight: float = 0.80,
        retrieval_weight: float = 0.20,
    ) -> None:
        if not isinstance(text_reranker, CrossEncoderReranker):
            raise TypeError("text_reranker must be a CrossEncoderReranker")
        if not isinstance(visual_embedder, BaseVisualEmbedder):
            raise TypeError("visual_embedder must be a BaseVisualEmbedder")
        if not 0.0 <= text_weight <= 1.0:
            raise ValueError("text_weight must be between 0 and 1")
        if not 0.0 <= retrieval_weight <= 1.0:
            raise ValueError("retrieval_weight must be between 0 and 1")
        if not np.isclose(text_weight + retrieval_weight, 1.0):
            raise ValueError("text_weight and retrieval_weight must sum to 1")

        self.text_reranker = text_reranker
        self.visual_embedder = visual_embedder
        self.text_weight = float(text_weight)
        self.retrieval_weight = float(retrieval_weight)

    @staticmethod
    def _normalize(values: Sequence[float]) -> List[float]:
        if not values:
            return []
        low = min(values)
        high = max(values)
        if high == low:
            return [0.5] * len(values)
        return [(value - low) / (high - low) for value in values]

    def rerank(
        self,
        query: str,
        candidates: Sequence[MultimodalSearchResult],
        top_k: Optional[int] = None,
    ) -> List[MultimodalSearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not candidates:
            return []
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")

        retrieval_scores = [
            float(candidate.retrieval_score
                  if candidate.retrieval_score is not None
                  else candidate.score)
            for candidate in candidates
        ]
        retrieval_norm = self._normalize(retrieval_scores)

        text_indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.element_type in self.TEXT_LIKE_TYPES
        ]
        text_scores: Dict[int, float] = {}
        if text_indices:
            pairs = [
                [query.strip(), default_candidate_text_extractor(candidates[index])]
                for index in text_indices
            ]
            try:
                raw_scores = self.text_reranker.model.predict(
                    pairs,
                    batch_size=self.text_reranker.batch_size,
                    show_progress_bar=False,
                )
            except TypeError:
                raw_scores = self.text_reranker.model.predict(pairs)
            raw_text_scores = self.text_reranker._normalize_scores(
                raw_scores, expected_count=len(text_indices)
            )
            normalized = self._normalize(raw_text_scores)
            for index, score in zip(text_indices, normalized):
                text_scores[index] = score

        image_indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.element_type == ElementType.IMAGE
        ]
        clip_scores: Dict[int, float] = {}
        if image_indices:
            query_vector = self.visual_embedder.embed_text_queries([query.strip()])[0]
            image_inputs = []
            for index in image_indices:
                element = candidates[index].element
                image_input = element.image_data
                if not image_input and element.image_path is not None:
                    image_input = element.image_path
                if not image_input:
                    image_input = element.content
                image_inputs.append(image_input)
            image_vectors = self.visual_embedder.embed_images(
                image_inputs
            )
            query_norm = np.linalg.norm(query_vector)
            image_norms = np.linalg.norm(image_vectors, axis=1)
            if query_norm == 0.0 or np.any(image_norms == 0.0):
                raise ValueError("Visual query or image embedding has zero norm")
            raw_clip = (image_vectors @ query_vector) / (
                image_norms * query_norm
            )
            normalized = self._normalize([float(score) for score in raw_clip])
            for index, score in zip(image_indices, normalized):
                clip_scores[index] = score

        scored: List[MultimodalSearchResult] = []
        for index, original in enumerate(candidates):
            is_text = index in text_scores
            is_image = index in clip_scores
            if not is_text and not is_image:
                raise ValueError(
                    f"Unsupported candidate element type: {original.element_type}"
                )
            modality_score = text_scores.get(index, clip_scores.get(index))
            if modality_score is None:
                raise ValueError(
                    f"No modality score was produced for '{original.element_id}'"
                )
            final_score = (
                self.text_weight * modality_score
                + self.retrieval_weight * retrieval_norm[index]
            )
            scored.append(
                MultimodalSearchResult(
                    element=original.element,
                    score=float(final_score),
                    element_id=original.element_id,
                    element_type=original.element_type,
                    source_filename=original.source_filename,
                    page_number=original.page_number,
                    related_element_ids=list(original.related_element_ids or []),
                    retrieval_score=retrieval_scores[index],
                    cross_encoder_score=text_scores.get(index),
                    clip_score=clip_scores.get(index),
                    final_score=float(final_score),
                )
            )

        scored.sort(key=lambda item: (-item.score, str(item.element_id or "")))
        return scored[:top_k] if top_k is not None else scored
