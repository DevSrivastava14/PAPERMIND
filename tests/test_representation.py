"""Tests for unified multimodal element representation layer (Day 12 Task 1)."""

import numpy as np
import pytest

from app.chunking.models import TextChunk
from app.multimodal import (
    ELEMENT_TYPE_CAPTION,
    ELEMENT_TYPE_EQUATION,
    ELEMENT_TYPE_IMAGE,
    ELEMENT_TYPE_TABLE,
    ELEMENT_TYPE_TEXT,
    ElementType,
    MultimodalElement,
    SUPPORTED_ELEMENT_TYPES,
    attach_caption_matches,
    from_caption,
    from_equation,
    from_image,
    from_table,
    from_text_chunk,
    to_multimodal_element,
)
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
# 1. Direct Creation of All Supported Element Types
# ===========================================================================


def test_multimodal_element_direct_creation_all_types():
    """Verify that MultimodalElement can be created for all supported element types."""
    types_and_contents = [
        (ElementType.TEXT, "Some text paragraph"),
        (ElementType.IMAGE, b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"),
        (ElementType.CAPTION, "Figure 1: System overview."),
        (ElementType.TABLE, "| Col1 | Col2 |\n|---|---|\n| Val1 | Val2 |"),
        (ElementType.EQUATION, "E = mc^2"),
    ]

    for elem_type, content in types_and_contents:
        el = MultimodalElement(
            element_id=f"doc.pdf#page_1_{elem_type.value}_0",
            element_type=elem_type,
            source_filename="doc.pdf",
            page_number=1,
            content=content,
        )
        assert el.element_type == elem_type
        assert el.element_type.value == elem_type.value
        assert el.source_filename == "doc.pdf"
        assert el.page_number == 1
        assert el.content == content
        assert el.embedding is None
        assert el.related_element_ids == []
        assert el.metadata == {}


def test_modality_boolean_properties():
    """Verify is_text, is_image, is_caption, is_table, is_equation flags."""
    text_el = MultimodalElement("id1", ElementType.TEXT, "doc.pdf", 1, "text")
    assert text_el.is_text is True
    assert text_el.is_image is False
    assert text_el.is_caption is False
    assert text_el.is_table is False
    assert text_el.is_equation is False

    img_el = MultimodalElement("id2", ElementType.IMAGE, "doc.pdf", 1, b"raw")
    assert img_el.is_image is True
    assert img_el.is_text is False

    cap_el = MultimodalElement("id3", ElementType.CAPTION, "doc.pdf", 1, "cap")
    assert cap_el.is_caption is True

    tbl_el = MultimodalElement("id4", ElementType.TABLE, "doc.pdf", 1, "tbl")
    assert tbl_el.is_table is True

    eq_el = MultimodalElement("id5", ElementType.EQUATION, "doc.pdf", 1, "eq")
    assert eq_el.is_equation is True


def test_element_type_enum_and_constants():
    """Verify ElementType values and module-level constants."""
    assert ElementType.TEXT.value == "text"
    assert ElementType.IMAGE.value == "image"
    assert ElementType.CAPTION.value == "caption"
    assert ElementType.TABLE.value == "table"
    assert ElementType.EQUATION.value == "equation"

    assert ELEMENT_TYPE_TEXT == "text"
    assert ELEMENT_TYPE_IMAGE == "image"
    assert ELEMENT_TYPE_CAPTION == "caption"
    assert ELEMENT_TYPE_TABLE == "table"
    assert ELEMENT_TYPE_EQUATION == "equation"

    assert SUPPORTED_ELEMENT_TYPES == {"text", "image", "caption", "table", "equation"}

    # String comparison works directly with ElementType
    assert ElementType.TEXT == "text"
    assert ElementType.IMAGE == "image"


def test_element_type_synonyms_normalization():
    """Verify common synonyms like figure, formula are mapped properly."""
    el1 = MultimodalElement("id1", "figure", "doc.pdf", 1, b"bytes")
    assert el1.element_type == ElementType.IMAGE

    el2 = MultimodalElement("id2", "formula", "doc.pdf", 1, "a^2 + b^2 = c^2")
    assert el2.element_type == ElementType.EQUATION


# ===========================================================================
# 2. Conversion from Specialized Models
# ===========================================================================


def test_conversion_from_text_chunk():
    """Verify conversion from TextChunk retains ID, provenance, and text content."""
    chunk = TextChunk(
        chunk_id="chunk_page_2_idx_3",
        text="This is a body paragraph extracted from page 2.",
        source_filename="research_paper.pdf",
        chunk_index=3,
        page_numbers=[2],
    )

    el = from_text_chunk(chunk)
    assert el.element_id == "chunk_page_2_idx_3"
    assert el.element_type == ElementType.TEXT
    assert el.source_filename == "research_paper.pdf"
    assert el.page_number == 2
    assert el.content == "This is a body paragraph extracted from page 2."
    assert el.text == "This is a body paragraph extracted from page 2."
    assert el.metadata["chunk_index"] == 3
    assert el.metadata["page_numbers"] == [2]


def test_conversion_from_text_chunk_override_page():
    """Verify page_number override for multi-page TextChunks."""
    chunk = TextChunk(
        chunk_id="c1",
        text="Multi-page chunk",
        source_filename="doc.pdf",
        chunk_index=0,
        page_numbers=[2, 3],
    )
    el = from_text_chunk(chunk, page_number=3)
    assert el.page_number == 3


def test_conversion_from_image():
    """Verify conversion from ExtractedImage retains ID, provenance, and binary payload."""
    img = ExtractedImage(
        image_id="paper.pdf#page_3_img_1",
        source_filename="paper.pdf",
        page_number=3,
        image_index=1,
        image_data=b"\x89PNG\r\n\x1a\ndata",
        image_format="png",
        width=300,
        height=200,
    )

    el = from_image(img)
    assert el.element_id == "paper.pdf#page_3_img_1"
    assert el.element_type == ElementType.IMAGE
    assert el.source_filename == "paper.pdf"
    assert el.page_number == 3
    assert el.content == b"\x89PNG\r\n\x1a\ndata"
    assert el.image_data == b"\x89PNG\r\n\x1a\ndata"
    assert el.image_bytes == b"\x89PNG\r\n\x1a\ndata"
    assert el.metadata["width"] == 300
    assert el.metadata["height"] == 200
    assert el.metadata["image_format"] == "png"


def test_conversion_from_image_path_only():
    """Verify conversion from ExtractedImage with path reference instead of bytes."""
    img = ExtractedImage(
        image_id="paper.pdf#page_1_img_0",
        source_filename="paper.pdf",
        page_number=1,
        image_index=0,
        image_path="/data/images/img0.png",
    )
    el = from_image(img)
    assert el.element_id == "paper.pdf#page_1_img_0"
    assert el.is_image is True
    assert el.image_path == "/data/images/img0.png"
    assert el.content == "/data/images/img0.png"


def test_conversion_from_caption():
    """Verify conversion from ExtractedCaption retains ID, provenance, and text."""
    cap = ExtractedCaption(
        caption_id="paper.pdf#page_4_cap_0",
        source_filename="paper.pdf",
        page_number=4,
        caption_index=0,
        caption_type=CAPTION_TYPE_FIGURE,
        text="Figure 4: Comparative evaluation across benchmarks.",
        label="Figure 4",
        bbox=(50.0, 400.0, 300.0, 420.0),
    )

    el = from_caption(cap)
    assert el.element_id == "paper.pdf#page_4_cap_0"
    assert el.element_type == ElementType.CAPTION
    assert el.source_filename == "paper.pdf"
    assert el.page_number == 4
    assert el.content == "Figure 4: Comparative evaluation across benchmarks."
    assert el.text == "Figure 4: Comparative evaluation across benchmarks."
    assert el.metadata["label"] == "Figure 4"
    assert el.metadata["caption_type"] == "figure"
    assert el.metadata["bbox"] == (50.0, 400.0, 300.0, 420.0)


def test_conversion_from_table():
    """Verify conversion from ExtractedTable retains ID, provenance, rows, and markdown."""
    tbl = ExtractedTable(
        table_id="paper.pdf#page_5_table_0",
        source_filename="paper.pdf",
        page_number=5,
        table_index=0,
        rows=[["Metric", "Score"], ["Accuracy", "0.95"], ["F1", "0.94"]],
        bbox=(72.0, 100.0, 400.0, 250.0),
    )

    el = from_table(tbl)
    assert el.element_id == "paper.pdf#page_5_table_0"
    assert el.element_type == ElementType.TABLE
    assert el.source_filename == "paper.pdf"
    assert el.page_number == 5
    assert el.content == tbl.content
    assert "| Metric | Score |" in el.content
    assert el.metadata["num_rows"] == 3
    assert el.metadata["num_cols"] == 2
    assert el.metadata["bbox"] == (72.0, 100.0, 400.0, 250.0)


def test_conversion_from_equation():
    """Verify conversion from ExtractedEquation retains ID, provenance, and equation text."""
    eq = ExtractedEquation(
        equation_id="paper.pdf#page_2_eq_0",
        source_filename="paper.pdf",
        page_number=2,
        equation_index=0,
        text="\\mathcal{L}_{total} = \\lambda_1 \\mathcal{L}_{reg} + \\lambda_2 \\mathcal{L}_{cls}",
        equation_label="(3)",
        bbox=(100.0, 200.0, 400.0, 230.0),
        is_display=True,
    )

    el = from_equation(eq)
    assert el.element_id == "paper.pdf#page_2_eq_0"
    assert el.element_type == ElementType.EQUATION
    assert el.source_filename == "paper.pdf"
    assert el.page_number == 2
    assert el.content == eq.text
    assert el.metadata["equation_label"] == "(3)"
    assert el.metadata["is_display"] is True
    assert el.metadata["bbox"] == (100.0, 200.0, 400.0, 230.0)


# ===========================================================================
# 3. Universal Converter and Classmethods
# ===========================================================================


def test_universal_converter_to_multimodal_element():
    """Verify to_multimodal_element auto-dispatches based on input type."""
    chunk = TextChunk("c1", "text", "doc.pdf", 0, [1])
    img = ExtractedImage("i1", "doc.pdf", 1, 0, b"data")
    cap = ExtractedCaption("cp1", "doc.pdf", 1, 0, CAPTION_TYPE_FIGURE, "Figure 1")
    tbl = ExtractedTable("t1", "doc.pdf", 1, 0, rows=[["a", "b"]])
    eq = ExtractedEquation("e1", "doc.pdf", 1, 0, text="x + y = z")

    el_chunk = to_multimodal_element(chunk)
    el_img = to_multimodal_element(img)
    el_cap = to_multimodal_element(cap)
    el_tbl = to_multimodal_element(tbl)
    el_eq = to_multimodal_element(eq)

    assert el_chunk.is_text
    assert el_img.is_image
    assert el_cap.is_caption
    assert el_tbl.is_table
    assert el_eq.is_equation

    # Identity passthrough
    assert to_multimodal_element(el_chunk) is el_chunk

    # Unsupported type
    with pytest.raises(TypeError, match="Cannot convert dict to MultimodalElement"):
        to_multimodal_element({"some": "dict"})


def test_multimodal_element_classmethods():
    """Verify classmethods on MultimodalElement work symmetrically."""
    chunk = TextChunk("c1", "text", "doc.pdf", 0, [1])
    el = MultimodalElement.from_text_chunk(chunk)
    assert el.is_text

    img = ExtractedImage("i1", "doc.pdf", 1, 0, b"data")
    el_img = MultimodalElement.from_image(img)
    assert el_img.is_image

    el_any = MultimodalElement.from_element(chunk)
    assert el_any.is_text


# ===========================================================================
# 4. Relationship Preservation and CaptionMatch Support
# ===========================================================================


def test_relationship_preservation_direct_ids():
    """Verify related element IDs can be passed directly and mutated."""
    el = MultimodalElement(
        element_id="paper.pdf#page_2_img_0",
        element_type=ElementType.IMAGE,
        source_filename="paper.pdf",
        page_number=2,
        content=b"img",
        related_element_ids=["paper.pdf#page_2_cap_0"],
    )
    assert el.related_element_ids == ["paper.pdf#page_2_cap_0"]
    assert el.relationships == ["paper.pdf#page_2_cap_0"]
    assert el.related_ids == ["paper.pdf#page_2_cap_0"]

    # Add another relationship
    el.add_relationship("paper.pdf#page_2_cap_1")
    assert el.related_element_ids == [
        "paper.pdf#page_2_cap_0",
        "paper.pdf#page_2_cap_1",
    ]

    # Deduplication
    el.add_relationship("paper.pdf#page_2_cap_0")
    assert len(el.related_element_ids) == 2


def test_relationship_via_caption_matches_in_conversion():
    """Verify conversion functions accept CaptionMatch instances directly."""
    img = ExtractedImage("img_1", "doc.pdf", 2, 0, b"data")
    cap = ExtractedCaption("cap_1", "doc.pdf", 2, 0, CAPTION_TYPE_FIGURE, "Figure 1")

    match = CaptionMatch(
        element_id="img_1",
        caption_id="cap_1",
        caption_type=CAPTION_TYPE_FIGURE,
        proximity_pts=15.0,
        source_filename="doc.pdf",
        page_number=2,
    )

    el_img = from_image(img, matches=[match])
    el_cap = from_caption(cap, matches=[match])

    assert el_img.related_element_ids == ["cap_1"]
    assert el_cap.related_element_ids == ["img_1"]


def test_attach_caption_matches_helper():
    """Verify attach_caption_matches links elements and captions bidirectionally."""
    img_el = MultimodalElement("img_0", ElementType.IMAGE, "paper.pdf", 1, b"raw")
    tbl_el = MultimodalElement("tbl_0", ElementType.TABLE, "paper.pdf", 2, "| A |")
    cap_fig = MultimodalElement("cap_fig", ElementType.CAPTION, "paper.pdf", 1, "Fig 1")
    cap_tbl = MultimodalElement("cap_tbl", ElementType.CAPTION, "paper.pdf", 2, "Tab 1")

    match1 = CaptionMatch("img_0", "cap_fig", CAPTION_TYPE_FIGURE, 12.0, "paper.pdf", 1)
    match2 = CaptionMatch("tbl_0", "cap_tbl", CAPTION_TYPE_TABLE, 20.0, "paper.pdf", 2)

    elements = [img_el, tbl_el, cap_fig, cap_tbl]
    attach_caption_matches(elements, [match1, match2])

    assert img_el.related_element_ids == ["cap_fig"]
    assert cap_fig.related_element_ids == ["img_0"]
    assert tbl_el.related_element_ids == ["cap_tbl"]
    assert cap_tbl.related_element_ids == ["tbl_0"]


# ===========================================================================
# 5. Validation Rules (Identifiers, Page Numbers, Types, Content)
# ===========================================================================


def test_validation_empty_element_id():
    """Verify empty or whitespace element_id raises ValueError."""
    with pytest.raises(ValueError, match="element_id cannot be empty"):
        MultimodalElement("", ElementType.TEXT, "doc.pdf", 1, "text")

    with pytest.raises(ValueError, match="element_id cannot be empty"):
        MultimodalElement("   ", ElementType.TEXT, "doc.pdf", 1, "text")


def test_validation_empty_source_filename():
    """Verify empty or whitespace source_filename raises ValueError."""
    with pytest.raises(ValueError, match="source_filename cannot be empty"):
        MultimodalElement("id", ElementType.TEXT, "", 1, "text")

    with pytest.raises(ValueError, match="source_filename cannot be empty"):
        MultimodalElement("id", ElementType.TEXT, "   ", 1, "text")


def test_validation_invalid_page_number():
    """Verify page_number must be an integer >= 1 (1-based)."""
    # 0 or negative
    with pytest.raises(ValueError, match="page_number must be an integer >= 1"):
        MultimodalElement("id", ElementType.TEXT, "doc.pdf", 0, "text")

    with pytest.raises(ValueError, match="page_number must be an integer >= 1"):
        MultimodalElement("id", ElementType.TEXT, "doc.pdf", -5, "text")

    # Wrong types (bool, str, float)
    with pytest.raises(ValueError, match="page_number must be an integer >= 1"):
        MultimodalElement("id", ElementType.TEXT, "doc.pdf", True, "text")  # type: ignore

    with pytest.raises(ValueError, match="page_number must be an integer >= 1"):
        MultimodalElement("id", ElementType.TEXT, "doc.pdf", "1", "text")  # type: ignore


def test_validation_invalid_element_type():
    """Verify unsupported element types raise ValueError."""
    for invalid_type in ["audio", "video", "graph", "unknown", ""]:
        with pytest.raises(ValueError, match="Invalid element_type"):
            MultimodalElement("id", invalid_type, "doc.pdf", 1, "content")


def test_validation_empty_content():
    """Verify None or empty content raises ValueError."""
    with pytest.raises(ValueError, match="content cannot be None"):
        MultimodalElement("id", ElementType.TEXT, "doc.pdf", 1, None)

    with pytest.raises(ValueError, match="content cannot be empty"):
        MultimodalElement("id", ElementType.TEXT, "doc.pdf", 1, "")

    with pytest.raises(ValueError, match="content cannot be empty"):
        MultimodalElement("id", ElementType.IMAGE, "doc.pdf", 1, b"")


def test_validation_embedding_and_relationships_types():
    """Verify type checking for embedding and relationships."""
    with pytest.raises(TypeError, match="embedding must be a list or ndarray"):
        MultimodalElement("id", ElementType.TEXT, "doc.pdf", 1, "text", embedding="invalid")  # type: ignore

    with pytest.raises(TypeError, match="related_element_ids must be a list"):
        MultimodalElement("id", ElementType.TEXT, "doc.pdf", 1, "text", related_element_ids="not_a_list")  # type: ignore


def test_to_dict_serialization():
    """Verify to_dict produces expected dictionary representation."""
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    el = MultimodalElement(
        element_id="el_1",
        element_type=ElementType.TEXT,
        source_filename="paper.pdf",
        page_number=1,
        content="Hello scientific RAG",
        embedding=vec,
        related_element_ids=["el_2"],
        metadata={"custom_key": "custom_val"},
    )

    d = el.to_dict()
    assert d["element_id"] == "el_1"
    assert d["element_type"] == "text"
    assert d["source_filename"] == "paper.pdf"
    assert d["page_number"] == 1
    assert d["content"] == "Hello scientific RAG"
    assert d["embedding"] == pytest.approx([0.1, 0.2, 0.3])
    assert d["related_element_ids"] == ["el_2"]
    assert d["metadata"]["custom_key"] == "custom_val"


def test_binary_content_to_dict_does_not_blow_up():
    """Verify to_dict with binary bytes content summarizes cleanly."""
    el = MultimodalElement(
        element_id="img_1",
        element_type=ElementType.IMAGE,
        source_filename="paper.pdf",
        page_number=1,
        content=b"12345678",
    )
    d = el.to_dict()
    assert d["content"] == "<8 bytes>"
