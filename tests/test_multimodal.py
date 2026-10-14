from pathlib import Path
import pytest
import pymupdf

from app.multimodal.image_extractor import ImageExtractor, extract_images
from app.multimodal.models import ExtractedImage, ExtractedVisualElement, VisualElement


def create_test_pdf_with_images(
    file_path: Path,
    pages_spec: list[dict],
) -> Path:
    """Helper to generate a PDF with custom text and embedded images per page.

    pages_spec is a list of dicts, each specifying:
      - 'text': optional str to draw
      - 'images': list of tuples (width, height, fill_val, rect_coords)
    """
    doc = pymupdf.open()
    for spec in pages_spec:
        page = doc.new_page()
        text = spec.get("text")
        if text:
            page.insert_text((72, 72), text)

        for img_info in spec.get("images", []):
            width, height, fill_val, rect_coords = img_info
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), 0)
            pix.clear_with(fill_val)
            img_bytes = pix.tobytes("png")
            page.insert_image(pymupdf.Rect(*rect_coords), stream=img_bytes)

    doc.save(str(file_path))
    doc.close()
    return file_path


def test_extract_images_single_image(tmp_path: Path):
    """Test extracting a single embedded image from a 1-page PDF."""
    pdf_path = tmp_path / "single_image.pdf"
    create_test_pdf_with_images(
        pdf_path,
        [
            {
                "text": "Page with an image",
                "images": [(20, 20, 200, (50, 50, 150, 150))],
            }
        ],
    )

    images = extract_images(pdf_path)

    assert len(images) == 1
    img = images[0]
    assert isinstance(img, ExtractedImage)
    assert img.source_filename == "single_image.pdf"
    assert img.page_number == 1
    assert img.image_index == 0
    assert img.image_id == "single_image.pdf#page_1_img_0"
    assert isinstance(img.image_data, bytes)
    assert len(img.image_data) > 0
    assert img.image_format == "png"
    assert img.width == 20
    assert img.height == 20
    assert img.image_path is None


def test_extract_images_correct_1_based_page_numbers(tmp_path: Path):
    """Verify exact 1-based page numbers when images appear on non-initial pages."""
    pdf_path = tmp_path / "page_provenance.pdf"
    create_test_pdf_with_images(
        pdf_path,
        [
            {"text": "Page 1 - Text only, no images", "images": []},
            {
                "text": "Page 2 - First figure",
                "images": [(15, 15, 100, (50, 50, 100, 100))],
            },
            {"text": "Page 3 - Text only", "images": []},
            {
                "text": "Page 4 - Second figure",
                "images": [(25, 25, 150, (50, 50, 120, 120))],
            },
        ],
    )

    images = extract_images(pdf_path)

    assert len(images) == 2
    assert images[0].page_number == 2
    assert images[0].image_index == 0
    assert images[0].image_id == "page_provenance.pdf#page_2_img_0"

    assert images[1].page_number == 4
    assert images[1].image_index == 0
    assert images[1].image_id == "page_provenance.pdf#page_4_img_0"


def test_extract_multiple_images_on_same_page(tmp_path: Path):
    """Verify distinct indexing and IDs for multiple images on the same page."""
    pdf_path = tmp_path / "multi_image_page.pdf"
    create_test_pdf_with_images(
        pdf_path,
        [
            {
                "text": "Page with two figures",
                "images": [
                    (10, 10, 255, (20, 20, 60, 60)),
                    (30, 30, 128, (80, 80, 150, 150)),
                ],
            }
        ],
    )

    images = extract_images(pdf_path)

    assert len(images) == 2
    assert images[0].page_number == 1
    assert images[0].image_index == 0
    assert images[0].image_id == "multi_image_page.pdf#page_1_img_0"

    assert images[1].page_number == 1
    assert images[1].image_index == 1
    assert images[1].image_id == "multi_image_page.pdf#page_1_img_1"

    assert images[0].image_id != images[1].image_id


def test_deterministic_image_identity_and_indexing(tmp_path: Path):
    """Verify that multiple extraction runs produce identical, deterministic results."""
    pdf_path = tmp_path / "deterministic.pdf"
    create_test_pdf_with_images(
        pdf_path,
        [
            {
                "text": "Page 1",
                "images": [(10, 10, 50, (10, 10, 50, 50))],
            },
            {
                "text": "Page 2",
                "images": [
                    (20, 20, 100, (10, 10, 60, 60)),
                    (30, 30, 150, (70, 70, 120, 120)),
                ],
            },
        ],
    )

    run_1 = extract_images(pdf_path)
    run_2 = extract_images(pdf_path)

    assert len(run_1) == len(run_2) == 3
    for img1, img2 in zip(run_1, run_2):
        assert img1.image_id == img2.image_id
        assert img1.source_filename == img2.source_filename
        assert img1.page_number == img2.page_number
        assert img1.image_index == img2.image_index
        assert img1.image_data == img2.image_data
        assert img1.image_format == img2.image_format
        assert img1.width == img2.width
        assert img1.height == img2.height


def test_extract_images_from_pdf_with_no_images(tmp_path: Path):
    """Handle text-only PDFs with no embedded images cleanly returning an empty list."""
    pdf_path = tmp_path / "no_images.pdf"
    create_test_pdf_with_images(
        pdf_path,
        [
            {"text": "Abstract text here", "images": []},
            {"text": "Methods text here", "images": []},
            {"text": "Conclusion text here", "images": []},
        ],
    )

    images = extract_images(pdf_path)

    assert isinstance(images, list)
    assert len(images) == 0


def test_extract_images_missing_file(tmp_path: Path):
    """Raise FileNotFoundError when PDF does not exist."""
    missing_path = tmp_path / "does_not_exist.pdf"

    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        extract_images(missing_path)


def test_extract_images_invalid_pdf(tmp_path: Path):
    """Raise ValueError when the file is corrupt or not a valid PDF."""
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_text("Not a real PDF binary content.")

    with pytest.raises(ValueError, match="Invalid PDF file"):
        extract_images(corrupt_path)


def test_extract_images_with_output_directory(tmp_path: Path):
    """Verify saving images to an output directory and setting image_path."""
    pdf_path = tmp_path / "with_output.pdf"
    create_test_pdf_with_images(
        pdf_path,
        [
            {
                "text": "Figure on Page 1",
                "images": [(16, 16, 200, (20, 20, 80, 80))],
            }
        ],
    )
    output_dir = tmp_path / "extracted_figures"

    images = extract_images(pdf_path, output_dir=output_dir)

    assert len(images) == 1
    img = images[0]
    assert img.image_path is not None
    saved_file = Path(img.image_path)
    assert saved_file.is_file()
    assert saved_file.read_bytes() == img.image_data


def test_image_extractor_class_interface(tmp_path: Path):
    """Verify ImageExtractor class instantiation and extract method."""
    pdf_path = tmp_path / "class_test.pdf"
    create_test_pdf_with_images(
        pdf_path,
        [
            {
                "text": "Class interface test",
                "images": [(12, 12, 180, (10, 10, 50, 50))],
            }
        ],
    )

    extractor = ImageExtractor()
    images = extractor.extract(pdf_path)

    assert len(images) == 1
    assert images[0].page_number == 1


def test_extracted_image_model_validation():
    """Verify validation rules in ExtractedImage dataclass."""
    # Valid instance
    img = ExtractedImage(
        image_id="doc.pdf#page_1_img_0",
        source_filename="doc.pdf",
        page_number=1,
        image_index=0,
        image_data=b"dummy_bytes",
    )
    assert img.image_bytes == b"dummy_bytes"
    assert img.path is None
    assert img.size == len(b"dummy_bytes")

    # Invalid page number (< 1)
    with pytest.raises(ValueError, match="page_number must be >= 1"):
        ExtractedImage("id", "doc.pdf", 0, 0, b"data")

    # Invalid image index (< 0)
    with pytest.raises(ValueError, match="image_index must be >= 0"):
        ExtractedImage("id", "doc.pdf", 1, -1, b"data")

    # Empty image_id
    with pytest.raises(ValueError, match="image_id cannot be empty"):
        ExtractedImage("", "doc.pdf", 1, 0, b"data")

    # Empty source_filename
    with pytest.raises(ValueError, match="source_filename cannot be empty"):
        ExtractedImage("id", "   ", 1, 0, b"data")

    # Non-bytes image_data
    with pytest.raises(TypeError, match="image_data must be bytes"):
        ExtractedImage("id", "doc.pdf", 1, 0, "string_data")  # type: ignore


def test_model_aliases():
    """Verify VisualElement and ExtractedVisualElement aliases."""
    assert VisualElement is ExtractedImage
    assert ExtractedVisualElement is ExtractedImage


def test_extract_images_sample_paper_smoke():
    """Smoke test on local sample paper fixture if present."""
    sample_path = Path("data/sample_paper.pdf")
    if not sample_path.exists():
        pytest.skip("Local sample paper fixture not present.")

    images = extract_images(sample_path)

    assert len(images) == 13
    assert all(isinstance(img, ExtractedImage) for img in images)
    assert all(img.page_number >= 1 for img in images)
    assert all(len(img.image_data) > 0 for img in images)
    assert all(img.source_filename == "sample_paper.pdf" for img in images)
    assert any(img.image_format == "jpeg" for img in images)
    assert any(img.image_format == "png" for img in images)
