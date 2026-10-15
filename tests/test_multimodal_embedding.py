"""Tests for the multimodal embedding layer (Day 12 Task 2)."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.embeddings.base import BaseEmbedder
from app.multimodal import (
    ElementType,
    MultimodalElement,
    from_caption,
    from_equation,
    from_image,
    from_table,
    from_text_chunk,
)
from app.multimodal.embedding import (
    BaseMultimodalEmbedder,
    BaseVisualEmbedder,
    MultimodalEmbeddedElement,
    MultimodalEmbedder,
    MultimodalEmbeddingResult,
    SentenceTransformerVisualEmbedder,
    embed_multimodal_elements,
    get_multimodal_embedder,
    reset_multimodal_embedder,
)
from app.multimodal.models import (
    CAPTION_TYPE_FIGURE,
    ExtractedCaption,
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
)


# ===========================================================================
# Helpers and Test Fixtures
# ===========================================================================


def _create_sample_png_bytes(color: str = "red", size: tuple[int, int] = (32, 32)) -> bytes:
    """Creates in-memory PNG bytes for testing visual encoding."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class MockTextEmbedder(BaseEmbedder):
    """Deterministic mock text embedder producing reproducible vectors."""

    def __init__(self, dimension: int = 384, model_name: str = "mock-text-embedder"):
        self._dim = dimension
        self._model_name = model_name
        self.call_count = 0

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        self.call_count += 1
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        # Create deterministic pseudo-embeddings based on text length
        vectors = []
        for idx, t in enumerate(texts):
            vec = np.full(self._dim, fill_value=(len(t) % 10 + 1) * 0.1, dtype=np.float32)
            vec[0] = float(idx + 1)
            vectors.append(vec)
        return np.vstack(vectors).astype(np.float32)


class MockVisualEmbedder(BaseVisualEmbedder):
    """Deterministic mock visual embedder producing reproducible vectors."""

    def __init__(self, dimension: int = 512, model_name: str = "mock-visual-embedder"):
        self._dim = dimension
        self._model_name = model_name
        self.call_count = 0

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_images(self, images: list[any], batch_size: int | None = None) -> np.ndarray:
        self.call_count += 1
        if not images:
            return np.empty((0, self._dim), dtype=np.float32)
        vectors = []
        for idx, _ in enumerate(images):
            vec = np.full(self._dim, fill_value=0.5, dtype=np.float32)
            vec[0] = float(idx + 100)
            vectors.append(vec)
        return np.vstack(vectors).astype(np.float32)


@pytest.fixture
def mock_text_embedder():
    return MockTextEmbedder(dimension=384)


@pytest.fixture
def mock_visual_embedder():
    return MockVisualEmbedder(dimension=512)


@pytest.fixture
def multimodal_embedder(mock_text_embedder, mock_visual_embedder):
    return MultimodalEmbedder(
        text_embedder=mock_text_embedder,
        visual_embedder=mock_visual_embedder,
    )


# ===========================================================================
# 1. Modality-Specific Embedding Tests
# ===========================================================================


def test_embed_text_element(multimodal_embedder):
    """Verify text element is routed to text embedder and preserves provenance."""
    el = MultimodalElement(
        element_id="paper.pdf#page_1_chunk_0",
        element_type=ElementType.TEXT,
        source_filename="paper.pdf",
        page_number=1,
        content="Deep learning representations for multimodal retrieval.",
    )

    res = multimodal_embedder.embed_element(el)
    assert isinstance(res, MultimodalEmbeddedElement)
    assert res.element_id == "paper.pdf#page_1_chunk_0"
    assert res.element_type == ElementType.TEXT
    assert res.source_filename == "paper.pdf"
    assert res.page_number == 1
    assert res.dimension == 384
    assert res.vector.shape == (384,)
    assert res.vector.dtype == np.float32
    assert res.is_text is True
    assert res.metadata["embedding_model"] == "mock-text-embedder"
    assert res.metadata["modality_stream"] == "text"
    assert el.embedding is not None


def test_embed_caption_element(multimodal_embedder):
    """Verify caption element is embedded using caption text."""
    cap = ExtractedCaption(
        caption_id="paper.pdf#page_2_cap_0",
        source_filename="paper.pdf",
        page_number=2,
        caption_index=0,
        caption_type=CAPTION_TYPE_FIGURE,
        text="Figure 1: High-level overview of PaperMind architecture.",
    )
    el = from_caption(cap)

    res = multimodal_embedder.embed_element(el)
    assert res.element_id == "paper.pdf#page_2_cap_0"
    assert res.element_type == ElementType.CAPTION
    assert res.dimension == 384
    assert res.is_caption is True
    assert res.metadata["modality_stream"] == "text"


def test_embed_table_element(multimodal_embedder):
    """Verify table element is embedded using its Markdown text representation."""
    tbl = ExtractedTable(
        table_id="paper.pdf#page_3_tbl_0",
        source_filename="paper.pdf",
        page_number=3,
        table_index=0,
        rows=[["Metric", "Score"], ["Precision", "0.91"], ["Recall", "0.89"]],
    )
    el = from_table(tbl)

    res = multimodal_embedder.embed_element(el)
    assert res.element_id == "paper.pdf#page_3_tbl_0"
    assert res.element_type == ElementType.TABLE
    assert res.dimension == 384
    assert res.is_table is True
    assert res.metadata["modality_stream"] == "text"


def test_embed_equation_element(multimodal_embedder):
    """Verify equation element is embedded using its formula string."""
    eq = ExtractedEquation(
        equation_id="paper.pdf#page_4_eq_0",
        source_filename="paper.pdf",
        page_number=4,
        equation_index=0,
        text="f(x) = \\sigma(W x + b)",
    )
    el = from_equation(eq)

    res = multimodal_embedder.embed_element(el)
    assert res.element_id == "paper.pdf#page_4_eq_0"
    assert res.element_type == ElementType.EQUATION
    assert res.dimension == 384
    assert res.is_equation is True
    assert res.metadata["modality_stream"] == "text"


def test_embed_image_element_bytes(multimodal_embedder):
    """Verify image element with raw binary bytes routes to visual embedder."""
    png_bytes = _create_sample_png_bytes(color="blue")
    img = ExtractedImage(
        image_id="paper.pdf#page_5_img_0",
        source_filename="paper.pdf",
        page_number=5,
        image_index=0,
        image_data=png_bytes,
    )
    el = from_image(img)

    res = multimodal_embedder.embed_element(el)
    assert res.element_id == "paper.pdf#page_5_img_0"
    assert res.element_type == ElementType.IMAGE
    assert res.dimension == 512
    assert res.vector.shape == (512,)
    assert res.is_image is True
    assert res.metadata["embedding_model"] == "mock-visual-embedder"
    assert res.metadata["modality_stream"] == "visual"


# ===========================================================================
# 2. Mixed-Modality Batches and 1:1 Mapping
# ===========================================================================


def test_mixed_modality_batch(multimodal_embedder, mock_text_embedder, mock_visual_embedder):
    """Verify a batch containing text, caption, table, equation, and image elements.

    Must preserve 1:1 mapping in exact input order, assign modality-appropriate
    dimensions, and update element.embedding.
    """
    png1 = _create_sample_png_bytes(color="green")
    png2 = _create_sample_png_bytes(color="yellow")

    elements = [
        MultimodalElement("el_txt1", ElementType.TEXT, "doc.pdf", 1, "Text chunk 1"),
        MultimodalElement("el_img1", ElementType.IMAGE, "doc.pdf", 1, png1),
        MultimodalElement("el_cap1", ElementType.CAPTION, "doc.pdf", 1, "Figure 1 caption"),
        MultimodalElement("el_tbl1", ElementType.TABLE, "doc.pdf", 2, "| A | B |\n|---|---|"),
        MultimodalElement("el_img2", ElementType.IMAGE, "doc.pdf", 2, png2),
        MultimodalElement("el_eq1", ElementType.EQUATION, "doc.pdf", 3, "a^2 + b^2 = c^2"),
    ]

    result = multimodal_embedder.embed_elements(elements)

    assert isinstance(result, MultimodalEmbeddingResult)
    assert len(result) == 6
    assert result.element_ids == [
        "el_txt1",
        "el_img1",
        "el_cap1",
        "el_tbl1",
        "el_img2",
        "el_eq1",
    ]

    # Verify each encoder was called once for the entire batch
    assert mock_text_embedder.call_count == 1
    assert mock_visual_embedder.call_count == 1

    # Verify dimensions for each slot match the respective modality
    assert result[0].dimension == 384  # text
    assert result[1].dimension == 512  # image
    assert result[2].dimension == 384  # caption
    assert result[3].dimension == 384  # table
    assert result[4].dimension == 512  # image
    assert result[5].dimension == 384  # equation

    # Check that original elements received embeddings
    for el in elements:
        assert el.embedding is not None


def test_element_id_to_vector_mapping(multimodal_embedder):
    """Verify lookup by element_id returns the exact vector or None."""
    elements = [
        MultimodalElement("id_a", ElementType.TEXT, "doc.pdf", 1, "alpha"),
        MultimodalElement("id_b", ElementType.IMAGE, "doc.pdf", 1, _create_sample_png_bytes()),
    ]
    result = multimodal_embedder.embed_elements(elements)

    vec_a = result.get_vector_by_id("id_a")
    assert vec_a is not None
    assert vec_a.shape == (384,)

    vec_b = result.get_vector_by_id("id_b")
    assert vec_b is not None
    assert vec_b.shape == (512,)

    assert result.get_vector_by_id("non_existent_id") is None
    assert result.get_by_element_id("non_existent_id") is None


def test_provenance_preservation(multimodal_embedder):
    """Verify source_filename and page_number provenance survive embedding."""
    elements = [
        MultimodalElement("e1", ElementType.TEXT, "biology.pdf", 7, "content 1"),
        MultimodalElement("e2", ElementType.IMAGE, "physics.pdf", 12, _create_sample_png_bytes()),
    ]
    result = multimodal_embedder.embed_elements(elements)

    ee1 = result.get_by_element_id("e1")
    assert ee1.source_filename == "biology.pdf"
    assert ee1.page_number == 7

    ee2 = result.get_by_element_id("e2")
    assert ee2.source_filename == "physics.pdf"
    assert ee2.page_number == 12


# ===========================================================================
# 3. Filtering and Matrix Conversion
# ===========================================================================


def test_get_vectors_by_type(multimodal_embedder):
    """Verify extracting vectors filtered by modality type."""
    elements = [
        MultimodalElement("t1", ElementType.TEXT, "doc.pdf", 1, "text 1"),
        MultimodalElement("i1", ElementType.IMAGE, "doc.pdf", 1, _create_sample_png_bytes()),
        MultimodalElement("t2", ElementType.TEXT, "doc.pdf", 2, "text 2"),
    ]
    result = multimodal_embedder.embed_elements(elements)

    text_matrix = result.get_vectors_by_type(ElementType.TEXT)
    assert text_matrix.shape == (2, 384)
    assert text_matrix.dtype == np.float32

    image_matrix = result.get_vectors_by_type(ElementType.IMAGE)
    assert image_matrix.shape == (1, 512)
    assert image_matrix.dtype == np.float32

    empty_matrix = result.get_vectors_by_type(ElementType.TABLE)
    assert empty_matrix.shape == (0, 0)


def test_to_numpy_matrix_homogeneous_and_mixed(multimodal_embedder):
    """Verify to_numpy_matrix succeeds on uniform batches and fails on mixed batches."""
    text_only = [
        MultimodalElement("t1", ElementType.TEXT, "doc.pdf", 1, "text 1"),
        MultimodalElement("t2", ElementType.CAPTION, "doc.pdf", 1, "caption 1"),
    ]
    res_text = multimodal_embedder.embed_elements(text_only)
    mat = res_text.to_numpy_matrix()
    assert mat.shape == (2, 384)

    mixed = [
        MultimodalElement("t1", ElementType.TEXT, "doc.pdf", 1, "text 1"),
        MultimodalElement("i1", ElementType.IMAGE, "doc.pdf", 1, _create_sample_png_bytes()),
    ]
    res_mixed = multimodal_embedder.embed_elements(mixed)
    with pytest.raises(ValueError, match="Cannot stack mixed-modality vectors with varying dimensions"):
        res_mixed.to_numpy_matrix()


# ===========================================================================
# 4. Visual Embedder Implementation and Validations
# ===========================================================================


def test_sentence_transformer_visual_embedder_mocked():
    """Verify SentenceTransformerVisualEmbedder decodes images and calls underlying model."""
    mock_st_model = MagicMock()
    # Mock model output: 2 images of dim 512
    mock_st_model.encode.return_value = np.ones((2, 512), dtype=np.float32)

    vis_embedder = SentenceTransformerVisualEmbedder(
        model=mock_st_model,
        dimension=512,
        model_name="test-clip",
    )

    png_bytes = _create_sample_png_bytes()
    pil_image = Image.new("RGB", (16, 16), color="red")

    vectors = vis_embedder.embed_images([png_bytes, pil_image])
    assert vectors.shape == (2, 512)
    assert vectors.dtype == np.float32
    mock_st_model.encode.assert_called_once()


def test_visual_embedder_file_path_input(tmp_path):
    """Verify SentenceTransformerVisualEmbedder can read image from a filesystem path."""
    img_path = tmp_path / "test_image.png"
    Image.new("RGB", (20, 20), color="green").save(img_path)

    mock_st_model = MagicMock()
    mock_st_model.encode.return_value = np.zeros((1, 512), dtype=np.float32)

    vis_embedder = SentenceTransformerVisualEmbedder(
        model=mock_st_model,
        dimension=512,
    )

    vectors = vis_embedder.embed_images([str(img_path)])
    assert vectors.shape == (1, 512)


def test_visual_embedder_empty_input():
    """Verify empty input returns array of shape (0, dimension)."""
    vis_embedder = SentenceTransformerVisualEmbedder(
        model=MagicMock(),
        dimension=512,
    )
    res = vis_embedder.embed_images([])
    assert res.shape == (0, 512)
    assert res.dtype == np.float32


def test_visual_embedder_invalid_bytes():
    """Verify corrupt or empty bytes raise ValueError."""
    vis_embedder = SentenceTransformerVisualEmbedder(
        model=MagicMock(),
        dimension=512,
    )
    with pytest.raises(ValueError, match="Image bytes at index 0 cannot be empty"):
        vis_embedder.embed_images([b""])

    with pytest.raises(ValueError, match="Failed to decode image bytes"):
        vis_embedder.embed_images([b"not_an_image"])


def test_visual_embedder_non_finite_values():
    """Verify vectors containing NaN or Inf raise ValueError."""
    mock_model = MagicMock()
    bad_vectors = np.ones((1, 512), dtype=np.float32)
    bad_vectors[0, 5] = np.nan
    mock_model.encode.return_value = bad_vectors

    vis_embedder = SentenceTransformerVisualEmbedder(
        model=mock_model,
        dimension=512,
    )
    png_bytes = _create_sample_png_bytes()

    with pytest.raises(ValueError, match="non-finite values"):
        vis_embedder.embed_images([png_bytes])


def test_visual_embedder_dimension_mismatch():
    """Verify mismatch between configured dimension and model output raises ValueError."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.ones((1, 256), dtype=np.float32)

    vis_embedder = SentenceTransformerVisualEmbedder(
        model=mock_model,
        dimension=512,
    )
    png_bytes = _create_sample_png_bytes()

    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        vis_embedder.embed_images([png_bytes])


def test_visual_embedder_vector_count_mismatch():
    """Verify mismatch in output vector count raises ValueError."""
    mock_model = MagicMock()
    # Provide 1 vector for 2 input images
    mock_model.encode.return_value = np.ones((1, 512), dtype=np.float32)

    vis_embedder = SentenceTransformerVisualEmbedder(
        model=mock_model,
        dimension=512,
    )
    png1 = _create_sample_png_bytes()
    png2 = _create_sample_png_bytes()

    with pytest.raises(ValueError, match="Vector count mismatch"):
        vis_embedder.embed_images([png1, png2])


# ===========================================================================
# 5. MultimodalEmbedder Edge Cases & Validation
# ===========================================================================


def test_empty_elements_input(multimodal_embedder):
    """Verify embedding empty element list returns empty result cleanly."""
    res = multimodal_embedder.embed_elements([])
    assert isinstance(res, MultimodalEmbeddingResult)
    assert len(res) == 0
    assert res.element_ids == []
    assert res.vectors == []


def test_invalid_element_in_batch(multimodal_embedder):
    """Verify non-MultimodalElement in list raises TypeError."""
    with pytest.raises(TypeError, match="must be a MultimodalElement"):
        multimodal_embedder.embed_elements(["string_not_element"])  # type: ignore


def test_unsupported_modality_handling(multimodal_embedder):
    """Verify unknown element modality raises ValueError."""
    el = MultimodalElement("id", ElementType.TEXT, "doc.pdf", 1, "test")
    # Simulate an invalid element_type attribute
    el.element_type = "unsupported_type"  # type: ignore

    with pytest.raises(ValueError, match="Unsupported element_type"):
        multimodal_embedder.embed_elements([el])


def test_embed_element_none_raises(multimodal_embedder):
    """Verify passing None to embed_element raises ValueError."""
    with pytest.raises(ValueError, match="element cannot be None"):
        multimodal_embedder.embed_element(None)


def test_convenience_function_and_singleton(mock_text_embedder, mock_visual_embedder):
    """Verify get_multimodal_embedder and embed_multimodal_elements convenience functions."""
    reset_multimodal_embedder()

    embedder = get_multimodal_embedder(
        text_embedder=mock_text_embedder,
        visual_embedder=mock_visual_embedder,
    )
    assert embedder.text_dimension == 384
    assert embedder.visual_dimension == 512

    el = MultimodalElement("c1", ElementType.TEXT, "doc.pdf", 1, "hello")
    res = embed_multimodal_elements([el], embedder=embedder)
    assert len(res) == 1
    assert res[0].element_id == "c1"

    reset_multimodal_embedder()
