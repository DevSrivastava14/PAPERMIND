from pathlib import Path
from typing import List, Optional
import pymupdf

from app.multimodal.models import ExtractedImage


class ImageExtractor:
    """Extracts embedded images and figures from research PDF pages using PyMuPDF."""

    def __init__(self, output_dir: Optional[str | Path] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else None

    def extract(self, pdf_path: str | Path) -> List[ExtractedImage]:
        """Extracts all embedded images from the given PDF.

        Preserves exact 1-based page provenance and deterministic image indexing.

        Args:
            pdf_path: File system path to the PDF document.

        Returns:
            List of ExtractedImage instances representing visual elements found.
            Returns an empty list if the PDF contains no images.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is not a valid or readable PDF.
        """
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")

        try:
            doc = pymupdf.open(path)
        except Exception as exc:
            raise ValueError(f"Invalid PDF file '{path}': {exc}") from exc

        extracted: List[ExtractedImage] = []
        try:
            for page_idx, page in enumerate(doc, start=1):
                try:
                    raw_images = page.get_images(full=True)
                except Exception:
                    raw_images = []

                for img_idx, img_info in enumerate(raw_images):
                    xref = img_info[0]
                    if xref <= 0:
                        continue

                    try:
                        base_image = doc.extract_image(xref)
                    except Exception:
                        continue

                    if not base_image or "image" not in base_image:
                        continue

                    image_bytes = base_image.get("image")
                    if not image_bytes or not isinstance(image_bytes, bytes):
                        continue

                    ext = base_image.get("ext", "png")
                    width = base_image.get("width")
                    height = base_image.get("height")

                    image_id = f"{path.name}#page_{page_idx}_img_{img_idx}"

                    saved_path_str: Optional[str] = None
                    if self.output_dir is not None:
                        self.output_dir.mkdir(parents=True, exist_ok=True)
                        safe_stem = path.stem
                        file_name = f"{safe_stem}_p{page_idx}_img{img_idx}.{ext}"
                        img_file = self.output_dir / file_name
                        img_file.write_bytes(image_bytes)
                        saved_path_str = str(img_file)

                    extracted.append(
                        ExtractedImage(
                            image_id=image_id,
                            source_filename=path.name,
                            page_number=page_idx,
                            image_index=img_idx,
                            image_data=image_bytes,
                            image_path=saved_path_str,
                            image_format=ext,
                            width=width,
                            height=height,
                        )
                    )
        except Exception as exc:
            raise ValueError(f"Failed to read images from PDF '{path}': {exc}") from exc
        finally:
            doc.close()

        return extracted


def extract_images(
    pdf_path: str | Path,
    output_dir: Optional[str | Path] = None,
) -> List[ExtractedImage]:
    """Convenience function to extract embedded images from a PDF.

    Args:
        pdf_path: File system path to the PDF document.
        output_dir: Optional directory where extracted images should be saved as files.

    Returns:
        List of ExtractedImage instances.
    """
    extractor = ImageExtractor(output_dir=output_dir)
    return extractor.extract(pdf_path)
