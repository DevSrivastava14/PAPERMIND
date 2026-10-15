from pathlib import Path
from typing import List, Optional
import pytest
import pymupdf

from app.multimodal import (
    ExtractedTable,
    ExtractedTableElement,
    TableElement,
    TableExtractor,
    extract_tables,
)


def _draw_table(
    page: pymupdf.Page,
    x0: float = 50.0,
    y0: float = 50.0,
    col_w: float = 100.0,
    row_h: float = 25.0,
    headers: Optional[List[str]] = None,
    data: Optional[List[List[str]]] = None,
) -> None:
    """Helper to draw a bordered table with lines and cell text on a PyMuPDF page."""
    headers = headers or ["Col1", "Col2"]
    data = data or [["Val1", "Val2"]]
    num_rows = 1 + len(data)
    num_cols = len(headers)

    # Draw horizontal grid lines
    for r in range(num_rows + 1):
        y = y0 + r * row_h
        page.draw_line((x0, y), (x0 + num_cols * col_w, y))

    # Draw vertical grid lines
    for c in range(num_cols + 1):
        x = x0 + c * col_w
        page.draw_line((x, y0), (x, y0 + num_rows * row_h))

    # Insert headers
    for c, h in enumerate(headers):
        page.insert_text((x0 + c * col_w + 5, y0 + 18), h)

    # Insert data rows
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            page.insert_text((x0 + c * col_w + 5, y0 + (r + 1) * row_h + 18), str(val))


def create_pdf_with_tables(
    file_path: Path,
    pages_spec: List[dict],
) -> Path:
    """Helper to generate a multi-page PDF with optional text and tables per page.

    pages_spec is a list of dicts:
      - 'text': optional str
      - 'tables': list of dicts with kwargs for _draw_table
    """
    doc = pymupdf.open()
    for spec in pages_spec:
        page = doc.new_page()
        if spec.get("text"):
            page.insert_text((50, 30), spec["text"])
        for tbl_kwargs in spec.get("tables", []):
            _draw_table(page, **tbl_kwargs)

    doc.save(str(file_path))
    doc.close()
    return file_path


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------


def test_extracted_table_init_valid():
    """Test valid construction and default field derivations."""
    rows = [["Method", "Accuracy"], ["PaperMind", "98.5%"], ["Baseline", "91.2%"]]
    tbl = ExtractedTable(
        table_id="paper.pdf#page_1_table_0",
        source_filename="paper.pdf",
        page_number=1,
        table_index=0,
        rows=rows,
        bbox=(50.0, 50.0, 250.0, 125.0),
    )

    assert tbl.table_id == "paper.pdf#page_1_table_0"
    assert tbl.source_filename == "paper.pdf"
    assert tbl.page_number == 1
    assert tbl.table_index == 0
    assert tbl.rows == rows
    assert tbl.num_rows == 3
    assert tbl.num_cols == 2
    assert tbl.bbox == (50.0, 50.0, 250.0, 125.0)
    assert tbl.headers == ["Method", "Accuracy"]
    assert "PaperMind" in tbl.content
    assert "Accuracy" in tbl.content


def test_extracted_table_properties_and_aliases():
    """Test property accessors and aliases for downstream flexibility."""
    rows = [["A", "B"], ["1", "2"]]
    tbl = ExtractedTable(
        table_id="doc.pdf#page_2_table_1",
        source_filename="doc.pdf",
        page_number=2,
        table_index=1,
        rows=rows,
    )

    # Properties
    assert tbl.row_count == 2
    assert tbl.col_count == 2
    assert tbl.cells == rows
    assert tbl.data == rows
    assert tbl.columns == ["A", "B"]
    assert tbl.table_content == tbl.content
    assert tbl.text == tbl.content
    assert tbl.markdown == tbl.content

    # Aliases
    assert TableElement is ExtractedTable
    assert ExtractedTableElement is ExtractedTable


def test_extracted_table_to_markdown_and_to_dict():
    """Test markdown generation and dict serialization."""
    rows = [["Model", "F1"], ["BERT", "0.89"]]
    tbl = ExtractedTable(
        table_id="test.pdf#page_1_table_0",
        source_filename="test.pdf",
        page_number=1,
        table_index=0,
        rows=rows,
        bbox=(10.0, 20.0, 100.0, 80.0),
    )

    md = tbl.to_markdown()
    assert "| Model | F1 |" in md
    assert "| --- | --- |" in md
    assert "| BERT | 0.89 |" in md

    d = tbl.to_dict()
    assert d["table_id"] == "test.pdf#page_1_table_0"
    assert d["source_filename"] == "test.pdf"
    assert d["page_number"] == 1
    assert d["table_index"] == 0
    assert d["num_rows"] == 2
    assert d["num_cols"] == 2
    assert d["rows"] == rows
    assert d["headers"] == ["Model", "F1"]
    assert d["bbox"] == (10.0, 20.0, 100.0, 80.0)


def test_extracted_table_validation_errors():
    """Test validation guards on ExtractedTable."""
    valid_rows = [["h1", "h2"], ["v1", "v2"]]

    # Empty table_id
    with pytest.raises(ValueError, match="table_id cannot be empty"):
        ExtractedTable(
            table_id="",
            source_filename="doc.pdf",
            page_number=1,
            table_index=0,
            rows=valid_rows,
        )

    # Whitespace table_id
    with pytest.raises(ValueError, match="table_id cannot be empty"):
        ExtractedTable(
            table_id="   ",
            source_filename="doc.pdf",
            page_number=1,
            table_index=0,
            rows=valid_rows,
        )

    # Empty source_filename
    with pytest.raises(ValueError, match="source_filename cannot be empty"):
        ExtractedTable(
            table_id="t1",
            source_filename="",
            page_number=1,
            table_index=0,
            rows=valid_rows,
        )

    # 0-based or negative page_number
    with pytest.raises(ValueError, match="page_number must be >= 1"):
        ExtractedTable(
            table_id="t1",
            source_filename="doc.pdf",
            page_number=0,
            table_index=0,
            rows=valid_rows,
        )

    # Negative table_index
    with pytest.raises(ValueError, match="table_index must be >= 0"):
        ExtractedTable(
            table_id="t1",
            source_filename="doc.pdf",
            page_number=1,
            table_index=-1,
            rows=valid_rows,
        )

    # Non-list rows
    with pytest.raises(TypeError, match="rows must be a list"):
        ExtractedTable(
            table_id="t1",
            source_filename="doc.pdf",
            page_number=1,
            table_index=0,
            rows="not a list",  # type: ignore
        )


# ---------------------------------------------------------------------------
# Extractor Error & Edge Case Tests
# ---------------------------------------------------------------------------


def test_extract_tables_missing_file_raises_filenotfound(tmp_path: Path):
    """Test that a non-existent file path raises FileNotFoundError."""
    missing_file = tmp_path / "non_existent.pdf"
    extractor = TableExtractor()
    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        extractor.extract(missing_file)


def test_extract_tables_invalid_file_raises_valueerror(tmp_path: Path):
    """Test that an invalid or corrupted file raises ValueError."""
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_text("This is not a valid PDF file content.")
    extractor = TableExtractor()
    with pytest.raises(ValueError, match="Invalid PDF file"):
        extractor.extract(corrupt_file)


def test_extract_tables_empty_pdf_returns_empty_list(tmp_path: Path):
    """Test that a valid PDF with no tables returns an empty list."""
    pdf_path = tmp_path / "text_only.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This page contains only standard prose and no tables.")
    doc.save(str(pdf_path))
    doc.close()

    tables = extract_tables(pdf_path)
    assert tables == []


def test_extract_tables_sample_paper_returns_empty_list():
    """Verify that data/sample_paper.pdf extracts cleanly with 0 tables."""
    sample_path = Path("data/sample_paper.pdf")
    assert sample_path.is_file(), "data/sample_paper.pdf must exist"

    tables = extract_tables(sample_path)
    assert isinstance(tables, list)
    assert len(tables) == 0


# ---------------------------------------------------------------------------
# Extraction Functionality & Provenance Tests
# ---------------------------------------------------------------------------


def test_extract_tables_single_table(tmp_path: Path):
    """Test detecting and extracting a single table from a 1-page PDF."""
    pdf_path = tmp_path / "single_table.pdf"
    create_pdf_with_tables(
        pdf_path,
        [
            {
                "text": "Table 1 below shows experiment results",
                "tables": [
                    {
                        "x0": 50.0,
                        "y0": 60.0,
                        "col_w": 100.0,
                        "row_h": 25.0,
                        "headers": ["Algorithm", "Speed"],
                        "data": [["A*", "12ms"], ["Dijkstra", "45ms"]],
                    }
                ],
            }
        ],
    )

    tables = extract_tables(pdf_path)
    assert len(tables) == 1

    tbl = tables[0]
    assert tbl.source_filename == "single_table.pdf"
    assert tbl.page_number == 1
    assert tbl.table_index == 0
    assert tbl.table_id == "single_table.pdf#page_1_table_0"
    assert tbl.num_rows >= 3
    assert tbl.num_cols == 2
    assert tbl.bbox is not None
    assert len(tbl.bbox) == 4
    assert tbl.headers is not None
    assert "Algorithm" in tbl.headers
    assert "Speed" in tbl.headers
    assert "A*" in tbl.content
    assert "Dijkstra" in tbl.content


def test_extract_tables_multi_page_provenance(tmp_path: Path):
    """Test extracting tables across multiple pages preserving exact 1-based page provenance."""
    pdf_path = tmp_path / "multi_page_tables.pdf"
    create_pdf_with_tables(
        pdf_path,
        [
            # Page 1: Text only
            {"text": "Introduction page with no tables."},
            # Page 2: Table A
            {
                "text": "Results page 2",
                "tables": [
                    {
                        "x0": 50.0,
                        "y0": 80.0,
                        "col_w": 80.0,
                        "row_h": 25.0,
                        "headers": ["Metric", "Value"],
                        "data": [["Precision", "0.91"]],
                    }
                ],
            },
            # Page 3: Table B
            {
                "text": "Results page 3",
                "tables": [
                    {
                        "x0": 50.0,
                        "y0": 80.0,
                        "col_w": 80.0,
                        "row_h": 25.0,
                        "headers": ["Hyperparam", "Setting"],
                        "data": [["LR", "0.001"]],
                    }
                ],
            },
        ],
    )

    tables = extract_tables(pdf_path)
    assert len(tables) == 2

    # First table provenance (Page 2)
    tbl0 = tables[0]
    assert tbl0.page_number == 2
    assert tbl0.table_index == 0
    assert tbl0.table_id == "multi_page_tables.pdf#page_2_table_0"
    assert "Precision" in tbl0.content

    # Second table provenance (Page 3)
    tbl1 = tables[1]
    assert tbl1.page_number == 3
    assert tbl1.table_index == 0
    assert tbl1.table_id == "multi_page_tables.pdf#page_3_table_0"
    assert "Hyperparam" in tbl1.content


def test_extract_tables_multiple_tables_same_page(tmp_path: Path):
    """Test extracting multiple tables on the same page with deterministic sequential indexing."""
    pdf_path = tmp_path / "two_tables_one_page.pdf"
    create_pdf_with_tables(
        pdf_path,
        [
            {
                "text": "Page with two separate tables",
                "tables": [
                    # Table 0 at top
                    {
                        "x0": 50.0,
                        "y0": 60.0,
                        "col_w": 90.0,
                        "row_h": 25.0,
                        "headers": ["Param", "Val"],
                        "data": [["Alpha", "0.5"]],
                    },
                    # Table 1 at bottom
                    {
                        "x0": 50.0,
                        "y0": 200.0,
                        "col_w": 90.0,
                        "row_h": 25.0,
                        "headers": ["Metric", "Val"],
                        "data": [["Recall", "0.88"]],
                    },
                ],
            }
        ],
    )

    tables = extract_tables(pdf_path)
    assert len(tables) == 2

    tbl0, tbl1 = tables[0], tables[1]
    assert tbl0.page_number == 1
    assert tbl0.table_index == 0
    assert tbl0.table_id == "two_tables_one_page.pdf#page_1_table_0"
    assert "Alpha" in tbl0.content

    assert tbl1.page_number == 1
    assert tbl1.table_index == 1
    assert tbl1.table_id == "two_tables_one_page.pdf#page_1_table_1"
    assert "Recall" in tbl1.content


def test_table_extractor_interface_parity(tmp_path: Path):
    """Test that TableExtractor().extract and extract_tables yield identical results."""
    pdf_path = tmp_path / "parity_test.pdf"
    create_pdf_with_tables(
        pdf_path,
        [
            {
                "tables": [
                    {
                        "x0": 50.0,
                        "y0": 50.0,
                        "col_w": 100.0,
                        "row_h": 25.0,
                        "headers": ["X", "Y"],
                        "data": [["10", "20"]],
                    }
                ]
            }
        ],
    )

    class_res = TableExtractor().extract(pdf_path)
    func_res = extract_tables(pdf_path)

    assert len(class_res) == len(func_res) == 1
    assert class_res[0].table_id == func_res[0].table_id
    assert class_res[0].rows == func_res[0].rows
    assert class_res[0].bbox == func_res[0].bbox
    assert class_res[0].content == func_res[0].content


def test_multimodal_module_exports():
    """Verify all table extraction symbols are properly exported from app.multimodal."""
    import app.multimodal as mm

    assert hasattr(mm, "ExtractedTable")
    assert hasattr(mm, "TableElement")
    assert hasattr(mm, "ExtractedTableElement")
    assert hasattr(mm, "TableExtractor")
    assert hasattr(mm, "extract_tables")

    assert mm.ExtractedTable is ExtractedTable
    assert mm.TableElement is ExtractedTable
    assert mm.ExtractedTableElement is ExtractedTable
    assert mm.TableExtractor is TableExtractor
    assert mm.extract_tables is extract_tables
