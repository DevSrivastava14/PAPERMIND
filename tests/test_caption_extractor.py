"""Focused Day 11 tests for caption extraction (app/multimodal/caption_extractor.py).

Structure
---------
* Model tests  – ExtractedCaption and CaptionMatch validation + accessors
* Extractor error tests  – missing file, invalid file, PDFs with no captions
* Extraction functionality tests  – figure captions, table captions, multi-page
  provenance, spatial matching, and interface parity
* Integration smoke tests  – sample_paper.pdf
* Module export test
"""
from pathlib import Path
from typing import List, Optional, Tuple
import pytest
import pymupdf

from app.multimodal import (
    CAPTION_TYPE_FIGURE,
    CAPTION_TYPE_TABLE,
    CaptionElement,
    CaptionExtractionResult,
    CaptionExtractor,
    CaptionMatch,
    ExtractedCaption,
    ExtractedCaptionElement,
    ExtractedTable,
    extract_captions,
)


# ===========================================================================
# Helpers to build synthetic PDFs
# ===========================================================================


def _add_table_to_page(
    page: pymupdf.Page,
    x0: float = 50.0,
    y0: float = 50.0,
    col_w: float = 80.0,
    row_h: float = 22.0,
    headers: Optional[List[str]] = None,
    data: Optional[List[List[str]]] = None,
) -> Tuple[float, float, float, float]:
    """Draw a simple bordered table; return its bounding box."""
    headers = headers or ["A", "B"]
    data = data or [["1", "2"]]
    n_rows = 1 + len(data)
    n_cols = len(headers)

    for r in range(n_rows + 1):
        y = y0 + r * row_h
        page.draw_line((x0, y), (x0 + n_cols * col_w, y))
    for c in range(n_cols + 1):
        x = x0 + c * col_w
        page.draw_line((x, y0), (x, y0 + n_rows * row_h))

    for c, h in enumerate(headers):
        page.insert_text((x0 + c * col_w + 5, y0 + 16), h)
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            page.insert_text((x0 + c * col_w + 5, y0 + (r + 1) * row_h + 16), str(val))

    return (x0, y0, x0 + n_cols * col_w, y0 + n_rows * row_h)


def _add_figure_block(
    page: pymupdf.Page,
    x0: float = 50.0,
    y0: float = 50.0,
    w: float = 150.0,
    h: float = 100.0,
    fill: int = 200,
) -> Tuple[float, float, float, float]:
    """Embed a solid-colour image block; return its bounding box."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, int(w), int(h)), 0)
    pix.clear_with(fill)
    img_bytes = pix.tobytes("png")
    rect = pymupdf.Rect(x0, y0, x0 + w, y0 + h)
    page.insert_image(rect, stream=img_bytes)
    return (x0, y0, x0 + w, y0 + h)


def _write_caption(
    page: pymupdf.Page,
    caption_text: str,
    x: float,
    y: float,
) -> None:
    """Insert a caption text at position (x, y)."""
    page.insert_text((x, y), caption_text)


def create_pdf_figure_caption(tmp_path: Path, caption: str = "Figure 1. A test figure.") -> Path:
    """One-page PDF: image block directly above, caption text below."""
    pdf_path = tmp_path / "figure_caption.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    _add_figure_block(page, x0=50, y0=50, w=150, h=100)
    # Caption immediately below the image
    _write_caption(page, caption, x=50, y=165)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def create_pdf_table_caption(
    tmp_path: Path,
    caption: str = "Table 1. Summary of results.",
) -> Path:
    """One-page PDF: table above, caption text below."""
    pdf_path = tmp_path / "table_caption.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    tbl_bbox = _add_table_to_page(page, x0=50, y0=50, headers=["X", "Y"], data=[["10", "20"]])
    # Caption just below the table
    _write_caption(page, caption, x=50, y=tbl_bbox[3] + 14)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def create_pdf_no_captions(tmp_path: Path) -> Path:
    """One-page PDF with plain text only — no captions."""
    pdf_path = tmp_path / "no_captions.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "This is plain prose with no figure or table captions.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def create_pdf_multi_page_captions(tmp_path: Path) -> Path:
    """Three-page PDF: page 1 has a figure caption, page 3 has a table caption."""
    pdf_path = tmp_path / "multi_page_captions.pdf"
    doc = pymupdf.open()

    # Page 1 – figure + figure caption
    p1 = doc.new_page()
    _add_figure_block(p1, x0=50, y0=50, w=150, h=100)
    _write_caption(p1, "Figure 1. First figure.", x=50, y=165)

    # Page 2 – plain text only
    p2 = doc.new_page()
    p2.insert_text((50, 50), "No captions here.")

    # Page 3 – table + table caption
    p3 = doc.new_page()
    tbl_bbox = _add_table_to_page(p3, x0=50, y0=50, headers=["M", "N"], data=[["a", "b"]])
    _write_caption(p3, "Table 1. Summary table.", x=50, y=tbl_bbox[3] + 14)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ===========================================================================
# Model Tests – ExtractedCaption
# ===========================================================================


class TestExtractedCaptionModel:
    def test_valid_figure_caption(self):
        cap = ExtractedCaption(
            caption_id="doc.pdf#page_2_cap_0",
            source_filename="doc.pdf",
            page_number=2,
            caption_index=0,
            caption_type="figure",
            text="Figure 1. A diagram of the architecture.",
            label="Figure 1",
            bbox=(43.0, 214.9, 271.1, 222.0),
        )
        assert cap.caption_id == "doc.pdf#page_2_cap_0"
        assert cap.source_filename == "doc.pdf"
        assert cap.page_number == 2
        assert cap.caption_index == 0
        assert cap.caption_type == CAPTION_TYPE_FIGURE
        assert cap.text == "Figure 1. A diagram of the architecture."
        assert cap.label == "Figure 1"
        assert cap.bbox == (43.0, 214.9, 271.1, 222.0)
        assert cap.is_figure_caption is True
        assert cap.is_table_caption is False
        assert cap.caption_text == cap.text
        assert cap.content == cap.text

    def test_valid_table_caption(self):
        cap = ExtractedCaption(
            caption_id="doc.pdf#page_3_cap_0",
            source_filename="doc.pdf",
            page_number=3,
            caption_index=0,
            caption_type="table",
            text="Table 2: Model comparison.",
            label="Table 2",
        )
        assert cap.caption_type == CAPTION_TYPE_TABLE
        assert cap.is_table_caption is True
        assert cap.is_figure_caption is False

    def test_caption_type_normalised_to_lowercase(self):
        cap = ExtractedCaption(
            caption_id="x.pdf#page_1_cap_0",
            source_filename="x.pdf",
            page_number=1,
            caption_index=0,
            caption_type="FIGURE",
            text="Figure 1.",
        )
        assert cap.caption_type == "figure"

    def test_to_dict(self):
        cap = ExtractedCaption(
            caption_id="a.pdf#page_1_cap_0",
            source_filename="a.pdf",
            page_number=1,
            caption_index=0,
            caption_type="figure",
            text="Figure 1. Test.",
            label="Figure 1",
            bbox=(0.0, 1.0, 2.0, 3.0),
        )
        d = cap.to_dict()
        assert d["caption_id"] == "a.pdf#page_1_cap_0"
        assert d["caption_type"] == "figure"
        assert d["text"] == "Figure 1. Test."
        assert d["label"] == "Figure 1"
        assert d["bbox"] == (0.0, 1.0, 2.0, 3.0)

    def test_aliases(self):
        assert CaptionElement is ExtractedCaption
        assert ExtractedCaptionElement is ExtractedCaption

    def test_validation_empty_caption_id(self):
        with pytest.raises(ValueError, match="caption_id cannot be empty"):
            ExtractedCaption(
                caption_id="",
                source_filename="a.pdf",
                page_number=1,
                caption_index=0,
                caption_type="figure",
                text="Figure 1.",
            )

    def test_validation_whitespace_caption_id(self):
        with pytest.raises(ValueError, match="caption_id cannot be empty"):
            ExtractedCaption(
                caption_id="   ",
                source_filename="a.pdf",
                page_number=1,
                caption_index=0,
                caption_type="figure",
                text="Figure 1.",
            )

    def test_validation_empty_source_filename(self):
        with pytest.raises(ValueError, match="source_filename cannot be empty"):
            ExtractedCaption(
                caption_id="a.pdf#page_1_cap_0",
                source_filename="",
                page_number=1,
                caption_index=0,
                caption_type="figure",
                text="Figure 1.",
            )

    def test_validation_page_number_zero(self):
        with pytest.raises(ValueError, match="page_number must be >= 1"):
            ExtractedCaption(
                caption_id="a.pdf#page_0_cap_0",
                source_filename="a.pdf",
                page_number=0,
                caption_index=0,
                caption_type="figure",
                text="Figure 1.",
            )

    def test_validation_negative_caption_index(self):
        with pytest.raises(ValueError, match="caption_index must be >= 0"):
            ExtractedCaption(
                caption_id="a.pdf#page_1_cap_0",
                source_filename="a.pdf",
                page_number=1,
                caption_index=-1,
                caption_type="figure",
                text="Figure 1.",
            )

    def test_validation_invalid_caption_type(self):
        with pytest.raises(ValueError, match="caption_type must be"):
            ExtractedCaption(
                caption_id="a.pdf#page_1_cap_0",
                source_filename="a.pdf",
                page_number=1,
                caption_index=0,
                caption_type="chart",
                text="Figure 1.",
            )

    def test_validation_empty_text(self):
        with pytest.raises(ValueError, match="caption text cannot be empty"):
            ExtractedCaption(
                caption_id="a.pdf#page_1_cap_0",
                source_filename="a.pdf",
                page_number=1,
                caption_index=0,
                caption_type="figure",
                text="",
            )


# ===========================================================================
# Model Tests – CaptionMatch
# ===========================================================================


class TestCaptionMatchModel:
    def test_valid_construction(self):
        m = CaptionMatch(
            element_id="doc.pdf#page_2_imgblock_3",
            caption_id="doc.pdf#page_2_cap_0",
            caption_type="figure",
            proximity_pts=12.6,
            source_filename="doc.pdf",
            page_number=2,
        )
        assert m.element_id == "doc.pdf#page_2_imgblock_3"
        assert m.caption_id == "doc.pdf#page_2_cap_0"
        assert m.caption_type == CAPTION_TYPE_FIGURE
        assert m.proximity_pts == pytest.approx(12.6)
        assert m.source_filename == "doc.pdf"
        assert m.page_number == 2

    def test_caption_type_normalised(self):
        m = CaptionMatch(
            element_id="e1",
            caption_id="c1",
            caption_type="TABLE",
            proximity_pts=5.0,
            source_filename="x.pdf",
            page_number=1,
        )
        assert m.caption_type == "table"

    def test_to_dict(self):
        m = CaptionMatch(
            element_id="e1",
            caption_id="c1",
            caption_type="figure",
            proximity_pts=8.0,
            source_filename="doc.pdf",
            page_number=3,
        )
        d = m.to_dict()
        assert d["element_id"] == "e1"
        assert d["caption_id"] == "c1"
        assert d["caption_type"] == "figure"
        assert d["proximity_pts"] == pytest.approx(8.0)

    def test_validation_empty_element_id(self):
        with pytest.raises(ValueError, match="element_id cannot be empty"):
            CaptionMatch("", "c1", "figure", 5.0, "doc.pdf", 1)

    def test_validation_empty_caption_id(self):
        with pytest.raises(ValueError, match="caption_id cannot be empty"):
            CaptionMatch("e1", "", "figure", 5.0, "doc.pdf", 1)

    def test_validation_invalid_type(self):
        with pytest.raises(ValueError, match="caption_type must be"):
            CaptionMatch("e1", "c1", "diagram", 5.0, "doc.pdf", 1)

    def test_validation_negative_proximity(self):
        with pytest.raises(ValueError, match="proximity_pts must be >= 0"):
            CaptionMatch("e1", "c1", "figure", -1.0, "doc.pdf", 1)

    def test_validation_page_number_zero(self):
        with pytest.raises(ValueError, match="page_number must be >= 1"):
            CaptionMatch("e1", "c1", "figure", 5.0, "doc.pdf", 0)


# ===========================================================================
# Extractor Error Tests
# ===========================================================================


class TestCaptionExtractorErrors:
    def test_missing_file_raises_filenotfound(self, tmp_path: Path):
        extractor = CaptionExtractor()
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            extractor.extract(tmp_path / "nonexistent.pdf")

    def test_invalid_file_raises_valueerror(self, tmp_path: Path):
        bad_file = tmp_path / "bad.pdf"
        bad_file.write_text("not a PDF")
        extractor = CaptionExtractor()
        with pytest.raises(ValueError, match="Invalid PDF file"):
            extractor.extract(bad_file)

    def test_missing_file_convenience_fn(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            extract_captions(tmp_path / "gone.pdf")


# ===========================================================================
# Extraction – No Captions
# ===========================================================================


class TestNoCaptions:
    def test_pdf_with_no_captions_returns_empty(self, tmp_path: Path):
        pdf = create_pdf_no_captions(tmp_path)
        result = extract_captions(pdf)
        assert isinstance(result, CaptionExtractionResult)
        assert result.captions == []
        assert result.matches == []

    def test_sample_paper_captions(self):
        """sample_paper.pdf has figure captions — smoke-test that they are found."""
        sample = Path("data/sample_paper.pdf")
        assert sample.is_file(), "data/sample_paper.pdf must exist"
        result = extract_captions(sample)
        assert isinstance(result.captions, list)
        # sample paper has 3 figure captions (verified in Day 11 probe)
        assert len(result.captions) == 3
        for cap in result.captions:
            assert cap.source_filename == "sample_paper.pdf"
            assert cap.page_number >= 1
            assert cap.caption_type == CAPTION_TYPE_FIGURE


# ===========================================================================
# Extraction – Figure Caption Detection & Provenance
# ===========================================================================


class TestFigureCaptionExtraction:
    def test_single_figure_caption_detected(self, tmp_path: Path):
        pdf = create_pdf_figure_caption(tmp_path, "Figure 1. A simple diagram.")
        result = extract_captions(pdf)
        assert len(result.captions) == 1
        cap = result.captions[0]
        assert cap.source_filename == "figure_caption.pdf"
        assert cap.page_number == 1
        assert cap.caption_index == 0
        assert cap.caption_type == CAPTION_TYPE_FIGURE
        assert "Figure 1" in cap.text
        assert cap.label == "Figure 1"
        assert cap.caption_id == "figure_caption.pdf#page_1_cap_0"
        assert cap.is_figure_caption is True

    def test_figure_caption_has_bbox(self, tmp_path: Path):
        pdf = create_pdf_figure_caption(tmp_path, "Figure 1. Bbox test.")
        result = extract_captions(pdf)
        assert result.captions
        cap = result.captions[0]
        assert cap.bbox is not None
        assert len(cap.bbox) == 4
        # bbox coords should be non-negative
        assert all(v >= 0 for v in cap.bbox)

    def test_figure_caption_spatially_matched_to_image(self, tmp_path: Path):
        pdf = create_pdf_figure_caption(tmp_path, "Figure 1. Diagram.")
        result = extract_captions(pdf)
        # The caption is within ~72 pts of the image block, so a match is expected
        assert len(result.matches) == 1
        m = result.matches[0]
        assert m.caption_type == CAPTION_TYPE_FIGURE
        assert m.caption_id == result.captions[0].caption_id
        assert m.proximity_pts >= 0
        assert m.page_number == 1

    def test_fig_prefix_recognised(self, tmp_path: Path):
        """'Fig. 1' should be detected as a figure caption."""
        pdf_path = tmp_path / "fig_prefix.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        _add_figure_block(page, x0=50, y0=50, w=150, h=100)
        _write_caption(page, "Fig. 1 – Architecture overview.", x=50, y=165)
        doc.save(str(pdf_path))
        doc.close()

        result = extract_captions(pdf_path)
        assert len(result.captions) == 1
        assert result.captions[0].caption_type == CAPTION_TYPE_FIGURE
        assert result.captions[0].label == "Figure 1"


# ===========================================================================
# Extraction – Table Caption Detection & Provenance
# ===========================================================================


class TestTableCaptionExtraction:
    def test_single_table_caption_detected(self, tmp_path: Path):
        pdf = create_pdf_table_caption(tmp_path, "Table 1. Summary of results.")
        result = extract_captions(pdf)
        assert len(result.captions) == 1
        cap = result.captions[0]
        assert cap.source_filename == "table_caption.pdf"
        assert cap.page_number == 1
        assert cap.caption_index == 0
        assert cap.caption_type == CAPTION_TYPE_TABLE
        assert "Table 1" in cap.text
        assert cap.label == "Table 1"
        assert cap.is_table_caption is True
        assert cap.caption_id == "table_caption.pdf#page_1_cap_0"

    def test_table_caption_matched_with_extracted_table(self, tmp_path: Path):
        """When ExtractedTable objects are supplied, captions should be matched."""
        pdf = create_pdf_table_caption(tmp_path, "Table 1. Results table.")
        result_no_tables = extract_captions(pdf)

        # Build a synthetic ExtractedTable with the known table bbox
        tbl = ExtractedTable(
            table_id="table_caption.pdf#page_1_table_0",
            source_filename="table_caption.pdf",
            page_number=1,
            table_index=0,
            rows=[["X", "Y"], ["10", "20"]],
            bbox=(50.0, 50.0, 210.0, 94.0),
        )
        result_with_tables = extract_captions(pdf, tables=[tbl])

        # With the table bbox supplied, the match should now appear
        assert len(result_with_tables.matches) == 1
        m = result_with_tables.matches[0]
        assert m.caption_type == CAPTION_TYPE_TABLE
        assert m.element_id == "table_caption.pdf#page_1_table_0"

    def test_tab_prefix_recognised(self, tmp_path: Path):
        """'Tab. 2' should be detected as a table caption."""
        pdf_path = tmp_path / "tab_prefix.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        tbl_bbox = _add_table_to_page(page, x0=50, y0=50)
        _write_caption(page, "Tab. 2 – Hyperparameter settings.", x=50, y=tbl_bbox[3] + 14)
        doc.save(str(pdf_path))
        doc.close()

        result = extract_captions(pdf_path)
        assert len(result.captions) == 1
        assert result.captions[0].caption_type == CAPTION_TYPE_TABLE
        assert result.captions[0].label == "Table 2"


# ===========================================================================
# Multi-page Provenance
# ===========================================================================


class TestMultiPageProvenance:
    def test_captions_have_correct_page_numbers(self, tmp_path: Path):
        pdf = create_pdf_multi_page_captions(tmp_path)
        result = extract_captions(pdf)
        assert len(result.captions) == 2

        # Figure caption should be on page 1
        fig_caps = [c for c in result.captions if c.caption_type == CAPTION_TYPE_FIGURE]
        assert len(fig_caps) == 1
        assert fig_caps[0].page_number == 1

        # Table caption should be on page 3
        tbl_caps = [c for c in result.captions if c.caption_type == CAPTION_TYPE_TABLE]
        assert len(tbl_caps) == 1
        assert tbl_caps[0].page_number == 3

    def test_deterministic_caption_ids(self, tmp_path: Path):
        pdf = create_pdf_multi_page_captions(tmp_path)
        result1 = extract_captions(pdf)
        result2 = extract_captions(pdf)
        ids1 = [c.caption_id for c in result1.captions]
        ids2 = [c.caption_id for c in result2.captions]
        assert ids1 == ids2


# ===========================================================================
# Interface Parity
# ===========================================================================


class TestInterfaceParity:
    def test_class_and_function_yield_same_result(self, tmp_path: Path):
        pdf = create_pdf_figure_caption(tmp_path, "Figure 1. Parity check.")
        class_result = CaptionExtractor().extract(pdf)
        func_result = extract_captions(pdf)
        assert len(class_result.captions) == len(func_result.captions)
        assert class_result.captions[0].caption_id == func_result.captions[0].caption_id
        assert class_result.captions[0].text == func_result.captions[0].text


# ===========================================================================
# Module Export Test
# ===========================================================================


class TestModuleExports:
    def test_all_caption_symbols_exported(self):
        import app.multimodal as mm

        assert hasattr(mm, "CAPTION_TYPE_FIGURE")
        assert hasattr(mm, "CAPTION_TYPE_TABLE")
        assert hasattr(mm, "ExtractedCaption")
        assert hasattr(mm, "CaptionElement")
        assert hasattr(mm, "ExtractedCaptionElement")
        assert hasattr(mm, "CaptionMatch")
        assert hasattr(mm, "CaptionExtractionResult")
        assert hasattr(mm, "CaptionExtractor")
        assert hasattr(mm, "extract_captions")

        assert mm.CAPTION_TYPE_FIGURE == "figure"
        assert mm.CAPTION_TYPE_TABLE == "table"
        assert mm.CaptionElement is ExtractedCaption
        assert mm.ExtractedCaptionElement is ExtractedCaption
        assert mm.CaptionExtractor is CaptionExtractor
        assert mm.extract_captions is extract_captions

    def test_existing_day8_10_exports_preserved(self):
        """Day 11 additions must not break existing exports."""
        import app.multimodal as mm

        for sym in [
            "ExtractedImage", "ExtractedVisualElement", "VisualElement",
            "ImageExtractor", "extract_images",
            "ExtractedEquation", "EquationElement", "ExtractedFormula",
            "EquationExtractor", "extract_equations",
            "ExtractedTable", "TableElement", "ExtractedTableElement",
            "TableExtractor", "extract_tables",
        ]:
            assert hasattr(mm, sym), f"Missing export: {sym}"
