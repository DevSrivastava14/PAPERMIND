from pathlib import Path
import pytest
import pymupdf

from app.multimodal.equation_extractor import EquationExtractor, extract_equations
from app.multimodal.models import EquationElement, ExtractedEquation, ExtractedFormula


def create_test_pdf_with_text(file_path: Path, pages_content: list[list[str]]) -> Path:
    """Helper to generate a PDF with multiple text blocks per page."""
    doc = pymupdf.open()
    for blocks in pages_content:
        page = doc.new_page()
        y_offset = 72
        for block_text in blocks:
            page.insert_text((72, y_offset), block_text)
            y_offset += 40
    doc.save(str(file_path))
    doc.close()
    return file_path


def test_extract_single_numbered_equation(tmp_path: Path):
    """Test extracting a single numbered equation from a 1-page PDF."""
    pdf_path = tmp_path / "single_eq.pdf"
    create_test_pdf_with_text(
        pdf_path,
        [
            [
                "We define the fundamental mass-energy equivalence as:",
                "E = m c^2                          (1)",
                "where E is energy, m is mass, and c is the speed of light.",
            ]
        ],
    )

    equations = extract_equations(pdf_path)

    assert len(equations) == 1
    eq = equations[0]
    assert isinstance(eq, ExtractedEquation)
    assert eq.source_filename == "single_eq.pdf"
    assert eq.page_number == 1
    assert eq.equation_index == 0
    assert eq.equation_id == "single_eq.pdf#page_1_eq_0"
    assert "E = m c^2" in eq.text
    assert eq.equation_label == "(1)"
    assert eq.is_display is True
    assert eq.bbox is not None
    assert len(eq.bbox) == 4


def test_extract_multiple_equations_on_same_page(tmp_path: Path):
    """Verify distinct sequential indexing and unique deterministic IDs on the same page."""
    pdf_path = tmp_path / "multi_eq_page.pdf"
    create_test_pdf_with_text(
        pdf_path,
        [
            [
                "First formulation:",
                "y = W x + b                          (1)",
                "Second formulation:",
                "f(x) = \\frac{1}{1 + e^{-x}}         (2)",
            ]
        ],
    )

    equations = extract_equations(pdf_path)

    assert len(equations) == 2
    assert equations[0].page_number == 1
    assert equations[0].equation_index == 0
    assert equations[0].equation_id == "multi_eq_page.pdf#page_1_eq_0"
    assert equations[0].equation_label == "(1)"

    assert equations[1].page_number == 1
    assert equations[1].equation_index == 1
    assert equations[1].equation_id == "multi_eq_page.pdf#page_1_eq_1"
    assert equations[1].equation_label == "(2)"

    assert equations[0].equation_id != equations[1].equation_id


def test_extract_equations_correct_1_based_page_numbers(tmp_path: Path):
    """Verify exact 1-based page numbers when equations appear on non-initial pages."""
    pdf_path = tmp_path / "page_provenance.pdf"
    create_test_pdf_with_text(
        pdf_path,
        [
            ["Page 1 is introduction text with no formulas."],
            [
                "Page 2 contains our first theorem:",
                "a^2 + b^2 = c^2                      (1)",
            ],
            ["Page 3 discusses empirical findings."],
            [
                "Page 4 presents the optimization objective:",
                "\\min_w \\frac{1}{2} ||w||_2^2       (2)",
            ],
        ],
    )

    equations = extract_equations(pdf_path)

    assert len(equations) == 2
    assert equations[0].page_number == 2
    assert equations[0].equation_index == 0
    assert equations[0].equation_id == "page_provenance.pdf#page_2_eq_0"

    assert equations[1].page_number == 4
    assert equations[1].equation_index == 0
    assert equations[1].equation_id == "page_provenance.pdf#page_4_eq_0"


def test_deterministic_equation_identity_and_indexing(tmp_path: Path):
    """Verify that multiple extraction runs produce identical, deterministic results."""
    pdf_path = tmp_path / "deterministic.pdf"
    create_test_pdf_with_text(
        pdf_path,
        [
            [
                "Intro text",
                "x_{t+1} = A x_t + B u_t              (1)",
                "Middle text",
                "y_t = C x_t + D u_t                  (2)",
            ]
        ],
    )

    run_1 = extract_equations(pdf_path)
    run_2 = extract_equations(pdf_path)

    assert len(run_1) == len(run_2) == 2
    for eq1, eq2 in zip(run_1, run_2):
        assert eq1.equation_id == eq2.equation_id
        assert eq1.source_filename == eq2.source_filename
        assert eq1.page_number == eq2.page_number
        assert eq1.equation_index == eq2.equation_index
        assert eq1.text == eq2.text
        assert eq1.equation_label == eq2.equation_label
        assert eq1.bbox == eq2.bbox


def test_extract_equations_from_pdf_with_no_equations(tmp_path: Path):
    """Handle text-only PDFs with no equations cleanly returning an empty list."""
    pdf_path = tmp_path / "no_equations.pdf"
    create_test_pdf_with_text(
        pdf_path,
        [
            ["Abstract: Generative AI is transforming enterprise product architecture."],
            ["Methods: We surveyed fifty technology firms regarding adoption."],
            ["Conclusion: Strategic governance remains essential."],
        ],
    )

    equations = extract_equations(pdf_path)

    assert isinstance(equations, list)
    assert len(equations) == 0


def test_extract_equations_missing_file(tmp_path: Path):
    """Raise FileNotFoundError when PDF does not exist."""
    missing_path = tmp_path / "does_not_exist.pdf"

    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        extract_equations(missing_path)


def test_extract_equations_invalid_pdf(tmp_path: Path):
    """Raise ValueError when the file is corrupt or not a valid PDF."""
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_text("Not a real PDF binary content.")

    with pytest.raises(ValueError, match="Invalid PDF file"):
        extract_equations(corrupt_path)


def test_equation_extractor_class_interface(tmp_path: Path):
    """Verify EquationExtractor class instantiation and extract method."""
    pdf_path = tmp_path / "class_test.pdf"
    create_test_pdf_with_text(
        pdf_path,
        [
            [
                "Theorem:",
                "e^{i \\pi} + 1 = 0                   (1)",
            ]
        ],
    )

    extractor = EquationExtractor()
    equations = extractor.extract(pdf_path)

    assert len(equations) == 1
    assert equations[0].page_number == 1
    assert equations[0].equation_label == "(1)"


def test_extracted_equation_model_validation():
    """Verify validation rules in ExtractedEquation dataclass."""
    # Valid instance
    eq = ExtractedEquation(
        equation_id="paper.pdf#page_1_eq_0",
        source_filename="paper.pdf",
        page_number=1,
        equation_index=0,
        text="E = m c^2",
        equation_label="(1)",
    )
    assert eq.equation_text == "E = m c^2"
    assert eq.expression == "E = m c^2"
    assert eq.content == "E = m c^2"

    # Invalid page number (< 1)
    with pytest.raises(ValueError, match="page_number must be >= 1"):
        ExtractedEquation("id", "doc.pdf", 0, 0, "x = 1")

    # Invalid equation index (< 0)
    with pytest.raises(ValueError, match="equation_index must be >= 0"):
        ExtractedEquation("id", "doc.pdf", 1, -1, "x = 1")

    # Empty equation_id
    with pytest.raises(ValueError, match="equation_id cannot be empty"):
        ExtractedEquation("", "doc.pdf", 1, 0, "x = 1")

    # Empty source_filename
    with pytest.raises(ValueError, match="source_filename cannot be empty"):
        ExtractedEquation("id", "   ", 1, 0, "x = 1")

    # Empty text
    with pytest.raises(ValueError, match="equation text cannot be empty"):
        ExtractedEquation("id", "doc.pdf", 1, 0, "")


def test_model_aliases():
    """Verify EquationElement and ExtractedFormula aliases."""
    assert EquationElement is ExtractedEquation
    assert ExtractedFormula is ExtractedEquation


def test_extract_equations_sample_paper_smoke():
    """Smoke test on local sample paper fixture confirming 0 equations detected cleanly."""
    sample_path = Path("data/sample_paper.pdf")
    if not sample_path.exists():
        pytest.skip("Local sample paper fixture not present.")

    equations = extract_equations(sample_path)

    # sample_paper.pdf is an enterprise AI survey paper without equations
    assert isinstance(equations, list)
    assert len(equations) == 0
