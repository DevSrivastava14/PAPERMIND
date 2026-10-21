from typing import List, Optional

import numpy as np
import pytest

from app.embeddings.base import BaseEmbedder
from app.indexing.multimodal_models import MultimodalSearchResult
from app.multimodal.embedding import BaseVisualEmbedder
from app.multimodal.representation import ElementType, MultimodalElement
from app.reranking.cross_encoder import CrossEncoderReranker
from app.reranking.multimodal import MultimodalReranker


class FakeTextEmbedder(BaseEmbedder):
    @property
    def dimension(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return "fake-text"

    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        return np.zeros((len(texts), 384), dtype=np.float32)


class SharedVisualEmbedder(BaseVisualEmbedder):
    @property
    def dimension(self) -> int:
        return 2

    @property
    def model_name(self) -> str:
        return "shared-test-visual"

    def embed_images(self, images, batch_size=None) -> np.ndarray:
        vectors = []
        for image in images:
            if image == b"good":
                vectors.append([1.0, 0.0])
            elif image == b"mid":
                vectors.append([0.5, 0.5])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)

    def embed_text_queries(self, texts, batch_size=None) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class UnsupportedVisualEmbedder(BaseVisualEmbedder):
    @property
    def dimension(self) -> int:
        return 2

    @property
    def model_name(self) -> str:
        return "image-only"

    def embed_images(self, images, batch_size=None) -> np.ndarray:
        return np.ones((len(images), 2), dtype=np.float32)


class FakeCrossEncoder:
    def __init__(self, scores=None):
        self.scores = scores

    def predict(self, pairs, **kwargs):
        if self.scores is not None:
            return list(self.scores)
        return [0.9 if "relevant" in pair[1].lower() else 0.1 for pair in pairs]


def candidate(element_id, element_type, content, score):
    element = MultimodalElement(
        element_id=element_id,
        element_type=element_type,
        source_filename="paper.pdf",
        page_number=1,
        content=content,
    )
    return MultimodalSearchResult(element=element, score=score)


@pytest.fixture
def reranker():
    return MultimodalReranker(
        text_reranker=CrossEncoderReranker(model=FakeCrossEncoder()),
        visual_embedder=SharedVisualEmbedder(),
    )


def test_text_and_caption_use_cross_encoder_and_preserve_components(reranker):
    results = reranker.rerank(
        "query",
        [
            candidate("text", ElementType.TEXT, "relevant text", 0.9),
            candidate("caption", ElementType.CAPTION, "ordinary caption", 0.8),
        ],
    )

    by_id = {result.element_id: result for result in results}
    assert by_id["text"].cross_encoder_score == 1.0
    assert by_id["text"].raw_cross_encoder_score == 0.9
    assert by_id["text"].clip_score is None
    assert by_id["caption"].cross_encoder_score == 0.0
    assert by_id["caption"].raw_cross_encoder_score == 0.1
    assert by_id["caption"].clip_score is None
    assert by_id["text"].retrieval_score == 0.9
    assert by_id["text"].final_score == by_id["text"].score


def test_image_uses_query_to_image_score_without_caption_inheritance(reranker):
    results = reranker.rerank(
        "query",
        [
            candidate("image-good", ElementType.IMAGE, b"good", 0.2),
            candidate("image-bad", ElementType.IMAGE, b"bad", 0.9),
        ],
    )

    assert [result.element_id for result in results] == ["image-good", "image-bad"]
    good, bad = results
    assert good.clip_score == 1.0
    assert bad.clip_score == 0.0
    assert good.raw_clip_score == 1.0
    assert bad.raw_clip_score == 0.0
    assert good.cross_encoder_score is None
    assert good.retrieval_score == 0.2
    assert good.score > bad.score


def test_raw_clip_score_is_preserved_separately_from_normalized_score(reranker):
    results = reranker.rerank(
        "query",
        [
            candidate("image-mid", ElementType.IMAGE, b"mid", 0.2),
            candidate("image-bad", ElementType.IMAGE, b"bad", 0.9),
        ],
    )

    mid = next(result for result in results if result.element_id == "image-mid")
    assert mid.raw_clip_score == pytest.approx(2**-0.5)
    assert mid.clip_score == pytest.approx(2**-0.5)


def test_unsupported_visual_text_query_fails_clearly():
    reranker = MultimodalReranker(
        text_reranker=CrossEncoderReranker(model=FakeCrossEncoder()),
        visual_embedder=UnsupportedVisualEmbedder(),
    )
    with pytest.raises(NotImplementedError, match="does not support text queries"):
        reranker.rerank(
            "query",
            [candidate("image", ElementType.IMAGE, b"image", 0.5)],
        )


def test_text_only_cross_encoder_behavior_remains_available():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    results = reranker.rerank(
        "query",
        [
            candidate("low", ElementType.TEXT, "ordinary text", 0.1),
            candidate("high", ElementType.TEXT, "relevant text", 0.2),
        ],
    )
    assert [result.element_id for result in results] == ["high", "low"]


def test_retrieval_normalization_retains_candidate_pool_min_max_behavior(reranker):
    assert reranker._normalize([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
    assert reranker._normalize([0.7, 0.7]) == [0.5, 0.5]


def test_ce_normalization_does_not_use_candidate_pool_endpoints():
    text_reranker = CrossEncoderReranker(
        model=FakeCrossEncoder(scores=[-8.6500, -8.6499, -8.6498])
    )
    reranker = MultimodalReranker(
        text_reranker=text_reranker,
        visual_embedder=SharedVisualEmbedder(),
    )
    results = reranker.rerank(
        "query",
        [
            candidate("ce-1", ElementType.TEXT, "one", 0.3),
            candidate("ce-2", ElementType.TEXT, "two", 0.2),
            candidate("ce-3", ElementType.TEXT, "three", 0.1),
        ],
    )
    by_id = {result.element_id: result for result in results}
    normalized = [by_id[f"ce-{index}"].cross_encoder_score for index in range(1, 4)]
    assert all(score is not None and 0.0 <= score <= 1.0 for score in normalized)
    assert normalized != [0.0, 0.5, 1.0]
    repeated = MultimodalReranker(
        text_reranker=CrossEncoderReranker(
            model=FakeCrossEncoder(scores=[-8.6500, -8.6499, -8.6498])
        ),
        visual_embedder=SharedVisualEmbedder(),
    ).rerank(
        "query",
        [
            candidate("ce-1", ElementType.TEXT, "one", 0.3),
            candidate("ce-2", ElementType.TEXT, "two", 0.2),
            candidate("ce-3", ElementType.TEXT, "three", 0.1),
        ],
    )
    repeated_by_id = {result.element_id: result for result in repeated}
    assert normalized == [
        repeated_by_id[f"ce-{index}"].cross_encoder_score
        for index in range(1, 4)
    ]


def test_ce_normalization_is_not_rebased_by_unrelated_extreme():
    def score_for_pool(scores):
        result = MultimodalReranker(
            text_reranker=CrossEncoderReranker(
                model=FakeCrossEncoder(scores=scores)
            ),
                visual_embedder=SharedVisualEmbedder(),
        ).rerank(
            "query",
            [
                candidate(f"ce-{index}", ElementType.TEXT, str(index), 0.3)
                for index in range(len(scores))
            ],
        )
        return next(item.cross_encoder_score for item in result if item.element_id == "ce-1")

    without_extreme = score_for_pool([-1.0, -0.5])
    with_extreme = score_for_pool([-100.0, -1.0, -0.5])
    assert without_extreme is not None and with_extreme is not None
    assert abs(without_extreme - with_extreme) < 0.2


def test_clip_single_candidate_normalization_uses_explicit_neutral_value(reranker):
    single = reranker.rerank(
        "query",
        [candidate("image", ElementType.IMAGE, b"good", 0.5)],
    )[0]
    paired = reranker.rerank(
        "query",
        [
            candidate("image", ElementType.IMAGE, b"good", 0.5),
            candidate("image-other", ElementType.IMAGE, b"bad", 0.4),
        ],
    )
    paired_image = next(item for item in paired if item.element_id == "image")
    assert single.clip_score is not None and 0.0 <= single.clip_score <= 1.0
    assert single.clip_score == 0.5
    assert paired_image.clip_score == 1.0


def test_clip_identical_scores_remain_tied(reranker):
    results = reranker.rerank(
        "query",
        [
            candidate("image-one", ElementType.IMAGE, b"good", 0.5),
            candidate("image-two", ElementType.IMAGE, b"good", 0.4),
        ],
    )
    scores = {result.element_id: result.clip_score for result in results}
    assert scores["image-one"] == scores["image-two"]
    assert scores["image-one"] is not None
    assert 0.0 <= scores["image-one"] <= 1.0


def test_multimodal_score_diagnostics_remain_preserved():
    reranker = MultimodalReranker(
        text_reranker=CrossEncoderReranker(
            model=FakeCrossEncoder(scores=[-2.0])
        ),
        visual_embedder=SharedVisualEmbedder(),
    )
    result = reranker.rerank(
        "query",
        [candidate("text", ElementType.TEXT, "text", 0.25)],
    )[0]
    assert result.raw_cross_encoder_score == -2.0
    assert result.raw_clip_score is None
    assert result.cross_encoder_score is not None
    assert result.clip_score is None
    assert result.retrieval_score == 0.25
    assert result.final_score == result.score
