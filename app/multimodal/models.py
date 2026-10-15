from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple



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


@dataclass
class ExtractedEquation:
    """Represents an extracted mathematical formula or equation from a PDF page.

    Attributes:
        equation_id: Unique deterministic identifier for the equation (e.g. 'paper.pdf#page_1_eq_0').
        source_filename: Name of the originating PDF document file.
        page_number: 1-based page number where the equation appeared.
        equation_index: 0-based index of the equation on the given page.
        text: Plain-text / LaTeX representation of the equation.
        equation_label: Optional label or number associated with the equation (e.g. '(1)', '(2.3)').
        bbox: Optional spatial bounding box (x0, y0, x1, y1) on the page in PDF points.
        is_display: True if the equation is a standalone display equation, False if inline.
    """

    equation_id: str
    source_filename: str
    page_number: int
    equation_index: int
    text: str = ""
    equation_label: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    is_display: bool = True

    def __init__(
        self,
        equation_id: str,
        source_filename: str,
        page_number: int,
        equation_index: int,
        text: Optional[str] = None,
        equation_label: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        is_display: bool = True,
        *,
        equation_text: Optional[str] = None,
        expression: Optional[str] = None,
    ) -> None:
        self.equation_id = equation_id
        self.source_filename = source_filename
        self.page_number = page_number
        self.equation_index = equation_index

        actual_text = (
            text
            if text is not None
            else (equation_text if equation_text is not None else expression)
        )
        self.text = actual_text if actual_text is not None else ""
        self.equation_label = equation_label
        self.bbox = bbox
        self.is_display = is_display

        self._validate()

    def _validate(self) -> None:
        if not self.equation_id or not str(self.equation_id).strip():
            raise ValueError("equation_id cannot be empty")
        if not self.source_filename or not str(self.source_filename).strip():
            raise ValueError("source_filename cannot be empty")
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1 (1-based), got {self.page_number}")
        if self.equation_index < 0:
            raise ValueError(f"equation_index must be >= 0, got {self.equation_index}")
        if not self.text or not str(self.text).strip():
            raise ValueError("equation text cannot be empty")

    @property
    def equation_text(self) -> str:
        """Alias for text."""
        return self.text

    @property
    def expression(self) -> str:
        """Alias for text."""
        return self.text

    @property
    def content(self) -> str:
        """Alias for text."""
        return self.text


EquationElement = ExtractedEquation
ExtractedFormula = ExtractedEquation


@dataclass
class ExtractedTable:
    """Represents an extracted structured table from a PDF page.

    Attributes:
        table_id: Unique deterministic identifier for the table (e.g. 'paper.pdf#page_1_table_0').
        source_filename: Name of the originating PDF document file.
        page_number: 1-based page number where the table appeared.
        table_index: 0-based index of the table on the given page.
        rows: 2D list of strings containing cell contents for each row.
        num_rows: Number of rows in the table.
        num_cols: Number of columns in the table.
        content: Formatted text/markdown representation of the table.
        bbox: Optional spatial bounding box (x0, y0, x1, y1) on the page in PDF points.
        headers: Optional list of column header strings.
    """

    table_id: str
    source_filename: str
    page_number: int
    table_index: int
    rows: List[List[str]] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    content: str = ""
    bbox: Optional[Tuple[float, float, float, float]] = None
    headers: Optional[List[str]] = None

    def __init__(
        self,
        table_id: str,
        source_filename: str,
        page_number: int,
        table_index: int,
        rows: Optional[List[List[str]]] = None,
        num_rows: Optional[int] = None,
        num_cols: Optional[int] = None,
        content: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        headers: Optional[List[str]] = None,
        *,
        table_content: Optional[str] = None,
        markdown: Optional[str] = None,
        text: Optional[str] = None,
        data: Optional[List[List[str]]] = None,
        columns: Optional[List[str]] = None,
    ) -> None:
        self.table_id = table_id
        self.source_filename = source_filename
        self.page_number = page_number
        self.table_index = table_index

        actual_rows = rows if rows is not None else (data if data is not None else [])
        if not isinstance(actual_rows, list):
            raise TypeError(f"rows must be a list of lists, got {type(actual_rows).__name__}")
        self.rows = [
            [str(cell).strip() if cell is not None else "" for cell in row]
            if isinstance(row, (list, tuple))
            else [str(row).strip()]
            for row in actual_rows
        ]


        self.num_rows = num_rows if num_rows is not None else len(self.rows)
        self.num_cols = (
            num_cols
            if num_cols is not None
            else (max((len(r) for r in self.rows), default=0))
        )

        actual_content = content or table_content or markdown or text
        if actual_content is not None and str(actual_content).strip():
            self.content = str(actual_content).strip()
        else:
            self.content = self._rows_to_markdown(self.rows)

        self.bbox = bbox

        actual_headers = headers if headers is not None else columns
        if actual_headers is not None:
            self.headers = [str(h).strip() for h in actual_headers]
        elif self.rows:
            self.headers = list(self.rows[0])
        else:
            self.headers = []

        self._validate()

    @staticmethod
    def _rows_to_markdown(rows: List[List[str]]) -> str:
        """Converts a 2D matrix of row cells into a standard markdown table string."""
        if not rows:
            return ""
        max_cols = max(len(r) for r in rows)
        if max_cols == 0:
            return ""

        padded_rows = [r + [""] * (max_cols - len(r)) for r in rows]
        lines = []
        header = padded_rows[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in range(max_cols)) + " |")
        for row in padded_rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def _validate(self) -> None:
        if not self.table_id or not str(self.table_id).strip():
            raise ValueError("table_id cannot be empty")
        if not self.source_filename or not str(self.source_filename).strip():
            raise ValueError("source_filename cannot be empty")
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1 (1-based), got {self.page_number}")
        if self.table_index < 0:
            raise ValueError(f"table_index must be >= 0, got {self.table_index}")
        if not isinstance(self.rows, list):
            raise TypeError(f"rows must be a list of lists, got {type(self.rows).__name__}")

    @property
    def table_content(self) -> str:
        """Alias for content."""
        return self.content

    @property
    def text(self) -> str:
        """Alias for content."""
        return self.content

    @property
    def markdown(self) -> str:
        """Alias for content."""
        return self.content

    @property
    def row_count(self) -> int:
        """Alias for num_rows."""
        return self.num_rows

    @property
    def col_count(self) -> int:
        """Alias for num_cols."""
        return self.num_cols

    @property
    def cells(self) -> List[List[str]]:
        """Alias for rows."""
        return self.rows

    @property
    def data(self) -> List[List[str]]:
        """Alias for rows."""
        return self.rows

    @property
    def columns(self) -> List[str]:
        """Header column names or empty list."""
        return self.headers if self.headers is not None else []

    def to_markdown(self) -> str:
        """Returns the table in Markdown format."""
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the table to a dictionary representation."""
        return {
            "table_id": self.table_id,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "rows": self.rows,
            "headers": self.headers,
            "content": self.content,
            "bbox": self.bbox,
        }


TableElement = ExtractedTable
ExtractedTableElement = ExtractedTable


# ---------------------------------------------------------------------------
# Caption types
# ---------------------------------------------------------------------------

#: Sentinel string for figure/image captions.
CAPTION_TYPE_FIGURE: str = "figure"
#: Sentinel string for table captions.
CAPTION_TYPE_TABLE: str = "table"


@dataclass
class ExtractedCaption:
    """Represents an extracted figure or table caption from a PDF page.

    Attributes:
        caption_id: Unique deterministic identifier (e.g. 'paper.pdf#page_2_cap_0').
        source_filename: Name of the originating PDF document file.
        page_number: 1-based page number where the caption appeared.
        caption_index: 0-based index of the caption on the given page.
        caption_type: Either ``CAPTION_TYPE_FIGURE`` ('figure') or ``CAPTION_TYPE_TABLE`` ('table').
        text: Full plain-text content of the caption.
        label: Short label parsed from the caption text (e.g. 'Figure 1', 'Table 2').
        bbox: Optional spatial bounding box (x0, y0, x1, y1) in PDF points.
    """

    caption_id: str
    source_filename: str
    page_number: int
    caption_index: int
    caption_type: str
    text: str
    label: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None

    def __init__(
        self,
        caption_id: str,
        source_filename: str,
        page_number: int,
        caption_index: int,
        caption_type: str,
        text: str,
        label: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        *,
        caption_text: Optional[str] = None,
        content: Optional[str] = None,
    ) -> None:
        self.caption_id = caption_id
        self.source_filename = source_filename
        self.page_number = page_number
        self.caption_index = caption_index
        self.caption_type = caption_type.strip().lower()

        actual_text = text if text is not None else (caption_text or content or "")
        self.text = str(actual_text).strip()

        self.label = label.strip() if label else None
        self.bbox = bbox

        self._validate()

    def _validate(self) -> None:
        if not self.caption_id or not str(self.caption_id).strip():
            raise ValueError("caption_id cannot be empty")
        if not self.source_filename or not str(self.source_filename).strip():
            raise ValueError("source_filename cannot be empty")
        if self.page_number < 1:
            raise ValueError(
                f"page_number must be >= 1 (1-based), got {self.page_number}"
            )
        if self.caption_index < 0:
            raise ValueError(f"caption_index must be >= 0, got {self.caption_index}")
        if self.caption_type not in (CAPTION_TYPE_FIGURE, CAPTION_TYPE_TABLE):
            raise ValueError(
                f"caption_type must be 'figure' or 'table', got '{self.caption_type}'"
            )
        if not self.text:
            raise ValueError("caption text cannot be empty")

    @property
    def is_figure_caption(self) -> bool:
        """True when this caption belongs to a figure or image."""
        return self.caption_type == CAPTION_TYPE_FIGURE

    @property
    def is_table_caption(self) -> bool:
        """True when this caption belongs to a table."""
        return self.caption_type == CAPTION_TYPE_TABLE

    @property
    def caption_text(self) -> str:
        """Alias for text."""
        return self.text

    @property
    def content(self) -> str:
        """Alias for text."""
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the caption to a dictionary representation."""
        return {
            "caption_id": self.caption_id,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "caption_index": self.caption_index,
            "caption_type": self.caption_type,
            "text": self.text,
            "label": self.label,
            "bbox": self.bbox,
        }


CaptionElement = ExtractedCaption
ExtractedCaptionElement = ExtractedCaption


# ---------------------------------------------------------------------------
# Lightweight caption-element relationship
# ---------------------------------------------------------------------------


@dataclass
class CaptionMatch:
    """Lightweight, deterministic link between a visual element and its caption.

    Attributes:
        element_id: The ``image_id`` or ``table_id`` of the matched visual element.
        caption_id: The ``caption_id`` of the matched caption.
        caption_type: Either 'figure' or 'table'.
        proximity_pts: Vertical distance in PDF points between the element and caption.
        source_filename: Name of the originating PDF document file.
        page_number: 1-based page number where both elements appear.
    """

    element_id: str
    caption_id: str
    caption_type: str
    proximity_pts: float
    source_filename: str
    page_number: int

    def __init__(
        self,
        element_id: str,
        caption_id: str,
        caption_type: str,
        proximity_pts: float,
        source_filename: str,
        page_number: int,
    ) -> None:
        self.element_id = element_id
        self.caption_id = caption_id
        self.caption_type = caption_type.strip().lower()
        self.proximity_pts = float(proximity_pts)
        self.source_filename = source_filename
        self.page_number = page_number

        self._validate()

    def _validate(self) -> None:
        if not self.element_id or not str(self.element_id).strip():
            raise ValueError("element_id cannot be empty")
        if not self.caption_id or not str(self.caption_id).strip():
            raise ValueError("caption_id cannot be empty")
        if self.caption_type not in (CAPTION_TYPE_FIGURE, CAPTION_TYPE_TABLE):
            raise ValueError(
                f"caption_type must be 'figure' or 'table', got '{self.caption_type}'"
            )
        if self.proximity_pts < 0:
            raise ValueError(f"proximity_pts must be >= 0, got {self.proximity_pts}")
        if not self.source_filename or not str(self.source_filename).strip():
            raise ValueError("source_filename cannot be empty")
        if self.page_number < 1:
            raise ValueError(
                f"page_number must be >= 1 (1-based), got {self.page_number}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the match to a dictionary representation."""
        return {
            "element_id": self.element_id,
            "caption_id": self.caption_id,
            "caption_type": self.caption_type,
            "proximity_pts": self.proximity_pts,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
        }
