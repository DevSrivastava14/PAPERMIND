from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ExtractedImage:
    """Represents an extracted visual element (figure, chart, image) from a PDF page.

    Attributes:
        image_id: Unique deterministic identifier for the extracted image (e.g. 'paper.pdf#page_1_img_0').
        source_filename: Name of the originating PDF document file.
        page_number: 1-based page number where the visual element appeared.
        image_index: 0-based index of the image on the given page.
        image_data: Raw binary byte content of the extracted image.
        image_path: Optional filesystem path where the image file is saved.
        image_format: Image file extension / format (e.g. 'png', 'jpeg').
        width: Optional width of the image in pixels.
        height: Optional height of the image in pixels.
    """

    image_id: str
    source_filename: str
    page_number: int
    image_index: int
    image_data: bytes = b""
    image_path: Optional[str] = None
    image_format: str = "png"
    width: Optional[int] = None
    height: Optional[int] = None

    def __init__(
        self,
        image_id: str,
        source_filename: str,
        page_number: int,
        image_index: int,
        image_data: Optional[bytes] = None,
        image_path: Optional[str | Path] = None,
        image_format: str = "png",
        width: Optional[int] = None,
        height: Optional[int] = None,
        *,
        image_bytes: Optional[bytes] = None,
        path: Optional[str | Path] = None,
    ) -> None:
        self.image_id = image_id
        self.source_filename = source_filename
        self.page_number = page_number
        self.image_index = image_index

        actual_bytes = image_data if image_data is not None else image_bytes
        self.image_data = actual_bytes if actual_bytes is not None else b""

        actual_path = image_path if image_path is not None else path
        self.image_path = str(actual_path) if actual_path is not None else None

        self.image_format = image_format
        self.width = width
        self.height = height

        self._validate()

    def _validate(self) -> None:
        if not self.image_id or not str(self.image_id).strip():
            raise ValueError("image_id cannot be empty")
        if not self.source_filename or not str(self.source_filename).strip():
            raise ValueError("source_filename cannot be empty")
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1 (1-based), got {self.page_number}")
        if self.image_index < 0:
            raise ValueError(f"image_index must be >= 0, got {self.image_index}")
        if not isinstance(self.image_data, bytes):
            raise TypeError(f"image_data must be bytes, got {type(self.image_data).__name__}")

    @property
    def image_bytes(self) -> bytes:
        """Alias for image_data."""
        return self.image_data

    @property
    def path(self) -> Optional[str]:
        """Alias for image_path."""
        return self.image_path

    @property
    def size(self) -> int:
        """Byte size of the image payload."""
        return len(self.image_data)


VisualElement = ExtractedImage
ExtractedVisualElement = ExtractedImage
