from pathlib import Path
import pytest
import pymupdf

from app.ingestion.models import PageContent, ResearchPaperDocument
from app.ingestion.pdf_parser import extract_pdf_pages, parse_research_paper


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


def test_parse_research_paper_structure_and_pages(tmp_path: Path):
    pdf_path = tmp_path / "quantum_mechanics.pdf"
    contents = ["Abstract and Introduction", "Methods and Results"]
    create_sample_pdf(pdf_path, contents)

    doc = parse_research_paper(pdf_path)

    assert isinstance(doc, ResearchPaperDocument)
    assert doc.filename == "quantum_mechanics.pdf"
    assert doc.total_pages == 2
    assert len(doc.pages) == 2

    for idx, expected_text in enumerate(contents, start=1):
        page = doc.pages[idx - 1]
        assert isinstance(page, PageContent)
        assert page.page_number == idx
        assert expected_text in page.text
        assert page.has_text is True


def test_parse_research_paper_with_blank_pages(tmp_path: Path):
    pdf_path = tmp_path / "doc_with_blanks.pdf"
    create_sample_pdf(pdf_path, ["Front matter", "", "   \t\n  "])

    doc = parse_research_paper(pdf_path)

    assert doc.total_pages == 3
    assert len(doc.pages) == 3

    assert doc.pages[0].has_text is True
    assert "Front matter" in doc.pages[0].text

    assert doc.pages[1].has_text is False
    assert doc.pages[1].text == ""

    assert doc.pages[2].has_text is False
    assert doc.pages[2].text.strip() == ""


def test_parse_research_paper_error_handling(tmp_path: Path):
    missing_file = tmp_path / "missing.pdf"
    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        parse_research_paper(missing_file)

    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_text("Not a real PDF.")
    with pytest.raises(ValueError, match="Invalid PDF file"):
        parse_research_paper(corrupt_file)


def test_parse_research_paper_sample_smoke():
    sample_path = Path("data/sample_paper.pdf")
    if not sample_path.exists():
        pytest.skip("Local sample paper fixture not present.")

    doc = parse_research_paper(sample_path)

    assert isinstance(doc, ResearchPaperDocument)
    assert doc.filename == "sample_paper.pdf"
    assert doc.total_pages == 7
    assert len(doc.pages) == 7
    assert all(isinstance(p, PageContent) for p in doc.pages)
    assert all(p.has_text for p in doc.pages)


def create_zero_page_pdf(file_path: Path) -> Path:
    """Helper to programmatically generate a valid PDF with zero pages."""
    content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
        b"xref\n0 3\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"trailer\n<< /Size 3 /Root 1 0 R >>\n"
        b"startxref\n115\n%%EOF\n"
    )
    file_path.write_bytes(content)
    return file_path


def test_zero_page_pdf(tmp_path: Path):
    zero_pdf = create_zero_page_pdf(tmp_path / "zero_page.pdf")

    with pytest.raises(ValueError, match="PDF contains no pages"):
        extract_pdf_pages(zero_pdf)

    with pytest.raises(ValueError, match="PDF contains no pages"):
        parse_research_paper(zero_pdf)


def test_zero_byte_pdf(tmp_path: Path):
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    with pytest.raises(ValueError, match="Invalid PDF file"):
        extract_pdf_pages(empty_file)

    with pytest.raises(ValueError, match="Invalid PDF file"):
        parse_research_paper(empty_file)


def test_unsupported_file_extension(tmp_path: Path):
    txt_file = tmp_path / "document.txt"
    txt_file.write_text("plain text content")

    with pytest.raises(ValueError, match=r"Unsupported file format '\.txt'"):
        extract_pdf_pages(txt_file)

    with pytest.raises(ValueError, match=r"Unsupported file format '\.txt'"):
        parse_research_paper(txt_file)


def test_oversized_pdf_raises_value_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PAPER_MIND_MAX_PDF_SIZE_MB", "0.0001")
    pdf_path = create_sample_pdf(
        tmp_path / "oversized.pdf",
        ["This page content exceeds the tiny configured size limit easily."],
    )

    with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
        extract_pdf_pages(pdf_path)

    with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
        parse_research_paper(pdf_path)




