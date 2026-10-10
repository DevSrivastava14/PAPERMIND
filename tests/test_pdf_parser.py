from pathlib import Path
import pytest
import pymupdf

from app.ingestion.pdf_parser import extract_pdf_pages


def create_sample_pdf(file_path: Path, pages_content: list[str]) -> Path:
    """Helper to programmatically generate a PDF with given content per page."""
    doc = pymupdf.open()
    for text in pages_content:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(str(file_path))
    doc.close()
    return file_path


def test_multipage_pdf(tmp_path: Path):
    pdf_path = tmp_path / "multipage.pdf"
    contents = ["Page 1: Introduction", "Page 2: Methodology", "Page 3: Conclusion"]
    create_sample_pdf(pdf_path, contents)

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 3
    for idx, expected_text in enumerate(contents, start=1):
        assert pages[idx - 1]["page_number"] == idx
        assert expected_text in pages[idx - 1]["text"]
        assert pages[idx - 1]["has_text"] is True


def test_page_numbering(tmp_path: Path):
    pdf_path = tmp_path / "numbered.pdf"
    create_sample_pdf(pdf_path, ["First", "Second", "Third", "Fourth"])

    pages = extract_pdf_pages(pdf_path)

    page_numbers = [p["page_number"] for p in pages]
    assert page_numbers == [1, 2, 3, 4]


def test_empty_page(tmp_path: Path):
    pdf_path = tmp_path / "with_empty_page.pdf"
    # Page 1 has text, Page 2 is completely blank, Page 3 has whitespace only
    create_sample_pdf(pdf_path, ["Has text", "", "   \n\t  "])

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 3

    assert pages[0]["has_text"] is True
    assert "Has text" in pages[0]["text"]

    assert pages[1]["has_text"] is False
    assert pages[1]["text"] == ""

    assert pages[2]["has_text"] is False
    assert pages[2]["text"].strip() == ""


def test_missing_file(tmp_path: Path):
    non_existent_file = tmp_path / "does_not_exist.pdf"

    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        extract_pdf_pages(non_existent_file)


def test_invalid_pdf(tmp_path: Path):
    corrupt_file = tmp_path / "invalid.pdf"
    corrupt_file.write_text("This is not a valid PDF binary content.")

    with pytest.raises(ValueError, match="Invalid PDF file"):
        extract_pdf_pages(corrupt_file)
