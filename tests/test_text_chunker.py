from pathlib import Path
import pytest

from app.chunking import TextChunk, TextChunker, chunk_document
from app.ingestion.models import PageContent, ResearchPaperDocument
from app.ingestion.pdf_parser import parse_research_paper


def create_mock_document(
    filename: str = "test_paper.pdf",
    pages: list[tuple[int, str, bool]] | None = None,
) -> ResearchPaperDocument:
    """Helper to construct in-memory ResearchPaperDocument instances."""
    if pages is None:
        pages = [(1, "Sample text content", True)]

    page_objs = [
        PageContent(page_number=num, text=txt, has_text=has_txt)
        for num, txt, has_txt in pages
    ]
    return ResearchPaperDocument(
        filename=filename,
        total_pages=len(page_objs),
        pages=page_objs,
    )


def test_chunk_document_basic():
    # 2,000 chars of predictable text
    text_content = "A" * 1200 + "B" * 800
    doc = create_mock_document(pages=[(1, text_content, True)])

    chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=150)

    # Total length: 2000. Step: 850.
    # Chunk 0: [0:1000]
    # Chunk 1: [850:1850]
    # Chunk 2: [1700:2000] (len 300)
    assert len(chunks) == 3

    assert chunks[0].chunk_id == "test_paper.pdf#chunk_0"
    assert chunks[0].chunk_index == 0
    assert len(chunks[0].text) == 1000
    assert chunks[0].source_filename == "test_paper.pdf"
    assert chunks[0].page_numbers == [1]

    assert chunks[1].chunk_id == "test_paper.pdf#chunk_1"
    assert chunks[1].chunk_index == 1
    assert len(chunks[1].text) == 1000
    # Overlap verification: last 150 chars of chunk 0 match first 150 chars of chunk 1
    assert chunks[0].text[-150:] == chunks[1].text[:150]

    assert chunks[2].chunk_id == "test_paper.pdf#chunk_2"
    assert chunks[2].chunk_index == 2
    assert len(chunks[2].text) == 300
    assert chunks[1].text[-150:] == chunks[2].text[:150]
    # Verify no final content is lost
    assert chunks[2].text == text_content[1700:2000]
    assert chunks[2].text.endswith("B" * 150)


def test_chunk_document_exact_chunk_size():
    doc = create_mock_document(pages=[(1, "X" * 1000, True)])
    chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=150)
    assert len(chunks) == 1
    assert len(chunks[0].text) == 1000
    assert chunks[0].text == "X" * 1000
    assert chunks[0].chunk_index == 0
    assert chunks[0].page_numbers == [1]


def test_chunk_document_short_text():
    short_text = "This is a brief abstract that fits within a single chunk."
    doc = create_mock_document(pages=[(1, short_text, True)])

    chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=150)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == short_text
    assert chunks[0].page_numbers == [1]
    assert chunks[0].chunk_id == "test_paper.pdf#chunk_0"


def test_chunk_document_blank_pages():
    doc = create_mock_document(
        pages=[
            (1, "", False),
            (2, "   \n\t  ", False),
            (3, "Valid content", True),
            (4, "", False),
        ]
    )

    chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=150)

    assert len(chunks) == 1
    assert chunks[0].page_numbers == [3]
    assert chunks[0].text == "Valid content"

    # Entirely blank document
    blank_doc = create_mock_document(pages=[(1, "", False), (2, "   ", False)])
    assert chunk_document(blank_doc) == []


def test_chunk_document_multipage_spanning():
    # Page 1: 600 characters of '1'
    # Page 2: 600 characters of '2'
    # Separator: "\n\n" (2 characters)
    # Total combined text: 600 + 2 + 600 = 1202 characters
    page_1_text = "1" * 600
    page_2_text = "2" * 600
    doc = create_mock_document(
        pages=[
            (1, page_1_text, True),
            (2, page_2_text, True),
        ]
    )

    chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=150, page_separator="\n\n")

    # Chunk 0: [0:1000] -> covers Page 1 (0..600), separator (600..602), and Page 2 (602..1000)
    # Chunk 1: [850:1202] -> starts in Page 2 (starts at 850, well past 602)
    assert len(chunks) == 2

    assert chunks[0].chunk_index == 0
    assert chunks[0].page_numbers == [1, 2]
    assert "1" * 600 in chunks[0].text
    assert "2" in chunks[0].text

    assert chunks[1].chunk_index == 1
    assert chunks[1].page_numbers == [2]
    assert "1" not in chunks[1].text
    assert "2" in chunks[1].text


def test_chunk_document_stable_id():
    doc = create_mock_document(
        filename="custom_experiment_report.pdf",
        pages=[(1, "A" * 1500, True)],
    )

    chunks_first = chunk_document(doc, chunk_size=500, chunk_overlap=100)
    chunks_second = chunk_document(doc, chunk_size=500, chunk_overlap=100)

    assert len(chunks_first) > 1
    assert len(chunks_first) == len(chunks_second)

    for i, chunk in enumerate(chunks_first):
        expected_id = f"custom_experiment_report.pdf#chunk_{i}"
        assert chunk.chunk_id == expected_id
        assert chunk.chunk_id == chunks_second[i].chunk_id


def test_chunk_document_invalid_parameters():
    doc = create_mock_document()

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_document(doc, chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_document(doc, chunk_size=-100, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        chunk_document(doc, chunk_size=500, chunk_overlap=-10)

    with pytest.raises(ValueError, match="chunk_overlap .* must be strictly less than chunk_size"):
        chunk_document(doc, chunk_size=500, chunk_overlap=500)

    with pytest.raises(ValueError, match="chunk_overlap .* must be strictly less than chunk_size"):
        chunk_document(doc, chunk_size=500, chunk_overlap=600)


def test_chunk_document_sample_paper_smoke():
    sample_path = Path("data/sample_paper.pdf")
    if not sample_path.exists():
        pytest.skip("Local sample paper fixture not found.")

    doc = parse_research_paper(sample_path)
    chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=150)

    assert len(chunks) > 0
    for idx, chunk in enumerate(chunks):
        assert isinstance(chunk, TextChunk)
        assert chunk.chunk_index == idx
        assert chunk.chunk_id == f"sample_paper.pdf#chunk_{idx}"
        assert chunk.source_filename == "sample_paper.pdf"
        assert len(chunk.text) > 0
        assert len(chunk.text) <= 1000
        assert len(chunk.page_numbers) >= 1
        assert all(1 <= p <= doc.total_pages for p in chunk.page_numbers)
