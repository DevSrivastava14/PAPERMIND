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
            vectors.append([1.0, 0.0] if image == b"good" else [0.0, 1.0])
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
    def predict(self, pairs, **kwargs):
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
    assert by_id["text"].clip_score is None
    assert by_id["caption"].cross_encoder_score == 0.0
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
    assert good.cross_encoder_score is None
    assert good.retrieval_score == 0.2
    assert good.score > bad.score


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
