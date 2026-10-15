"""Integration tests for multimodal collection and assembly layer (Day 12 Task 3)."""

import io
import numpy as np
import pytest
from PIL import Image

from app.chunking.models import TextChunk
from app.embeddings.base import BaseEmbedder
from app.multimodal import (
    ElementType,
    MultimodalCollection,
    MultimodalDocument,
    MultimodalElement,
    MultimodalEmbedder,
    assemble_multimodal_collection,
    assemble_multimodal_document,
)
from app.multimodal.embedding import BaseVisualEmbedder
from app.multimodal.models import (
    CAPTION_TYPE_FIGURE,
    CAPTION_TYPE_TABLE,
    CaptionMatch,
    ExtractedCaption,
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
)


# ===========================================================================
# Test Helpers
# ===========================================================================


def _create_sample_png_bytes(color: str = "red", size: tuple[int, int] = (16, 16)) -> bytes:
    """Helper creating in-memory PNG bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class MockTextEmbedder(BaseEmbedder):
    """Deterministic 384-d mock text embedder."""

    @property
    def dimension(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return "mock-minilm-384"

    def embed_texts(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        vectors = []
        for idx, t in enumerate(texts):
            vec = np.full(384, fill_value=0.1 * (idx + 1), dtype=np.float32)
            vec[0] = float(len(t))
            vectors.append(vec)
        return np.vstack(vectors).astype(np.float32)


class MockVisualEmbedder(BaseVisualEmbedder):
    """Deterministic 512-d mock visual embedder."""

    @property
    def dimension(self) -> int:
        return 512

    @property
    def model_name(self) -> str:
        return "mock-clip-512"

    def embed_images(self, images: list[any], batch_size: int | None = None) -> np.ndarray:
        if not images:
            return np.empty((0, 512), dtype=np.float32)
        vectors = []
        for idx, _ in enumerate(images):
            vec = np.full(512, fill_value=0.2 * (idx + 1), dtype=np.float32)
            vec[0] = float(idx + 100)
            vectors.append(vec)
        return np.vstack(vectors).astype(np.float32)


@pytest.fixture
def mock_multimodal_embedder():
    return MultimodalEmbedder(
        text_embedder=MockTextEmbedder(),
        visual_embedder=MockVisualEmbedder(),
    )


# ===========================================================================
# Integration Tests
# ===========================================================================


def test_assembly_realistic_paper_components():
    """Verify assembling all extracted scientific-paper components into a coherent document."""
    # 1. Text chunk
    chunk = TextChunk(
        chunk_id="paper.pdf#chunk_0",
        text="Recent advancements in multimodal representations have unified textual and visual modalities.",
        source_filename="paper.pdf",
        chunk_index=0,
        page_numbers=[1],
    )

    # 2. Image
    img_bytes = _create_sample_png_bytes(color="cyan")
    image = ExtractedImage(
        image_id="paper.pdf#page_2_img_0",
        source_filename="paper.pdf",
        page_number=2,
        image_index=0,
        image_data=img_bytes,
        width=400,
        height=300,
    )

    # 3. Figure caption
    fig_caption = ExtractedCaption(
        caption_id="paper.pdf#page_2_cap_0",
        source_filename="paper.pdf",
        page_number=2,
        caption_index=0,
        caption_type=CAPTION_TYPE_FIGURE,
        text="Figure 1: Multimodal pipeline architecture.",
        label="Figure 1",
    )

    # 4. Table
    table = ExtractedTable(
        table_id="paper.pdf#page_3_table_0",
        source_filename="paper.pdf",
        page_number=3,
        table_index=0,
        rows=[["Model", "F1"], ["Baseline", "84.2"], ["Ours", "91.8"]],
    )

    # 5. Table caption
    table_caption = ExtractedCaption(
        caption_id="paper.pdf#page_3_cap_0",
        source_filename="paper.pdf",
        page_number=3,
        caption_index=0,
        caption_type=CAPTION_TYPE_TABLE,
        text="Table 1: Benchmark comparison against baseline models.",
        label="Table 1",
    )

    # 6. Equation
    equation = ExtractedEquation(
        equation_id="paper.pdf#page_4_eq_0",
        source_filename="paper.pdf",
        page_number=4,
        equation_index=0,
        text="\\mathcal{L}_{align} = -\\sum_{i} \\log \\frac{\\exp(s_{ii}/\\tau)}{\\sum_j \\exp(s_{ij}/\\tau)}",
        equation_label="(1)",
    )

    # 7. Day 11 CaptionMatch relationships
    matches = [
        CaptionMatch(
            element_id="paper.pdf#page_2_img_0",
            caption_id="paper.pdf#page_2_cap_0",
            caption_type=CAPTION_TYPE_FIGURE,
            proximity_pts=14.0,
            source_filename="paper.pdf",
            page_number=2,
        ),
        CaptionMatch(
            element_id="paper.pdf#page_3_table_0",
            caption_id="paper.pdf#page_3_cap_0",
            caption_type=CAPTION_TYPE_TABLE,
            proximity_pts=8.5,
            source_filename="paper.pdf",
            page_number=3,
        ),
    ]

    # Assemble document
    doc = assemble_multimodal_document(
        text_chunks=[chunk],
        images=[image],
        captions=[fig_caption, table_caption],
        tables=[table],
        equations=[equation],
        caption_matches=matches,
        source_filename="paper.pdf",
    )

    assert isinstance(doc, MultimodalDocument)
    assert len(doc) == 6
    assert doc.source_filename == "paper.pdf"

    # ID stability and uniqueness
    expected_ids = [
        "paper.pdf#chunk_0",
        "paper.pdf#page_2_img_0",
        "paper.pdf#page_2_cap_0",
        "paper.pdf#page_3_cap_0",
        "paper.pdf#page_3_table_0",
        "paper.pdf#page_4_eq_0",
    ]
    assert doc.element_ids == expected_ids
    assert len(set(doc.element_ids)) == 6

    # Verify element types
    assert doc["paper.pdf#chunk_0"].element_type == ElementType.TEXT
    assert doc["paper.pdf#page_2_img_0"].element_type == ElementType.IMAGE
    assert doc["paper.pdf#page_2_cap_0"].element_type == ElementType.CAPTION
    assert doc["paper.pdf#page_3_cap_0"].element_type == ElementType.CAPTION
    assert doc["paper.pdf#page_3_table_0"].element_type == ElementType.TABLE
    assert doc["paper.pdf#page_4_eq_0"].element_type == ElementType.EQUATION

    # Verify provenance
    assert doc["paper.pdf#chunk_0"].page_number == 1
    assert doc["paper.pdf#page_2_img_0"].page_number == 2
    assert doc["paper.pdf#page_2_cap_0"].page_number == 2
    assert doc["paper.pdf#page_3_table_0"].page_number == 3
    assert doc["paper.pdf#page_3_cap_0"].page_number == 3
    assert doc["paper.pdf#page_4_eq_0"].page_number == 4

    # Verify relationships: image <-> caption
    related_to_img = doc.get_related("paper.pdf#page_2_img_0")
    assert len(related_to_img) == 1
    assert related_to_img[0].element_id == "paper.pdf#page_2_cap_0"

    related_to_fig_cap = doc.get_related("paper.pdf#page_2_cap_0")
    assert len(related_to_fig_cap) == 1
    assert related_to_fig_cap[0].element_id == "paper.pdf#page_2_img_0"

    # Verify relationships: table <-> caption
    related_to_tbl = doc.get_related("paper.pdf#page_3_table_0")
    assert len(related_to_tbl) == 1
    assert related_to_tbl[0].element_id == "paper.pdf#page_3_cap_0"

    related_to_tbl_cap = doc.get_related("paper.pdf#page_3_cap_0")
    assert len(related_to_tbl_cap) == 1
    assert related_to_tbl_cap[0].element_id == "paper.pdf#page_3_table_0"

    # Element without relationships returns empty list
    assert doc.get_related("paper.pdf#chunk_0") == []
    assert doc.get_related("paper.pdf#page_4_eq_0") == []


def test_collection_query_operations():
    """Verify get_by_id, get_by_type, get_by_page, and dict serialization."""
    el1 = MultimodalElement("t1", ElementType.TEXT, "doc.pdf", 1, "text p1")
    el2 = MultimodalElement("c1", ElementType.CAPTION, "doc.pdf", 1, "caption p1")
    el3 = MultimodalElement("t2", ElementType.TEXT, "doc.pdf", 2, "text p2")
    el4 = MultimodalElement("i1", ElementType.IMAGE, "doc.pdf", 2, _create_sample_png_bytes())

    doc = MultimodalDocument([el1, el2, el3, el4])

    # get_by_id
    assert doc.get_by_id("t1") is el1
    assert doc.get_by_id("missing") is None

    # get_by_type
    texts = doc.get_by_type(ElementType.TEXT)
    assert len(texts) == 2
    assert texts[0].element_id == "t1"
    assert texts[1].element_id == "t2"

    images = doc.get_by_type("image")
    assert len(images) == 1
    assert images[0].element_id == "i1"

    # get_by_page
    p1_elements = doc.get_by_page(1)
    assert [e.element_id for e in p1_elements] == ["t1", "c1"]

    p2_elements = doc.get_by_page(2)
    assert [e.element_id for e in p2_elements] == ["t2", "i1"]

    assert doc.get_by_page(99) == []

    with pytest.raises(ValueError, match="page_number must be an integer >= 1"):
        doc.get_by_page(0)

    # __contains__ and __getitem__
    assert "t1" in doc
    assert "missing" not in doc
    assert doc[0] is el1
    assert doc["c1"] is el2

    with pytest.raises(KeyError):
        _ = doc["missing"]

    # to_dict
    d = doc.to_dict()
    assert d["element_count"] == 4
    assert d["element_ids"] == ["t1", "c1", "t2", "i1"]


def test_collection_embedding_integration(mock_multimodal_embedder):
    """Verify document.embed associates embeddings correctly and preserves separate vector spaces."""
    el_text = MultimodalElement("txt", ElementType.TEXT, "doc.pdf", 1, "Hello world")
    el_cap = MultimodalElement("cap", ElementType.CAPTION, "doc.pdf", 1, "Figure 1")
    el_img = MultimodalElement("img", ElementType.IMAGE, "doc.pdf", 1, _create_sample_png_bytes())
    el_tbl = MultimodalElement("tbl", ElementType.TABLE, "doc.pdf", 2, "| A | B |")
    el_eq = MultimodalElement("eq", ElementType.EQUATION, "doc.pdf", 2, "E = mc^2")

    doc = MultimodalDocument([el_text, el_cap, el_img, el_tbl, el_eq])

    assert doc.is_fully_embedded is False
    assert len(doc.embedded_elements()) == 0

    # Execute embedding
    res = doc.embed(embedder=mock_multimodal_embedder)

    assert len(res) == 5
    assert doc.is_fully_embedded is True
    assert len(doc.embedded_elements()) == 5

    # Check vector spaces remain separate and correct
    vec_text = doc.get_embedding("txt")
    assert vec_text is not None
    assert vec_text.shape == (384,)
    assert vec_text.dtype == np.float32

    vec_cap = doc.get_embedding("cap")
    assert vec_cap is not None
    assert vec_cap.shape == (384,)

    vec_tbl = doc.get_embedding("tbl")
    assert vec_tbl is not None
    assert vec_tbl.shape == (384,)

    vec_eq = doc.get_embedding("eq")
    assert vec_eq is not None
    assert vec_eq.shape == (384,)

    vec_img = doc.get_embedding("img")
    assert vec_img is not None
    assert vec_img.shape == (512,)
    assert vec_img.dtype == np.float32

    # Check modality-specific matrix extraction
    text_matrix = doc.get_vectors_by_type(ElementType.TEXT)
    assert text_matrix.shape == (1, 384)

    image_matrix = doc.get_vectors_by_type(ElementType.IMAGE)
    assert image_matrix.shape == (1, 512)

    # Missing ID embedding
    assert doc.get_embedding("non_existent") is None


def test_duplicate_element_id_rejection():
    """Verify adding an element with duplicate ID raises ValueError."""
    el1 = MultimodalElement("dup_id", ElementType.TEXT, "doc.pdf", 1, "first")
    el2 = MultimodalElement("dup_id", ElementType.TEXT, "doc.pdf", 1, "second")

    doc = MultimodalDocument([el1])
    with pytest.raises(ValueError, match="Duplicate element_id detected"):
        doc.add_element(el2)


def test_empty_document_assembly():
    """Verify assembling with no elements produces an empty document cleanly."""
    doc = assemble_multimodal_document()
    assert isinstance(doc, MultimodalDocument)
    assert len(doc) == 0
    assert doc.element_ids == []
    assert doc.source_filename is None


def test_alias_equivalence():
    """Verify MultimodalCollection and assemble_multimodal_collection aliases."""
    assert MultimodalCollection is MultimodalDocument
    assert assemble_multimodal_collection is assemble_multimodal_document
