"""Caption extraction for research PDFs using PyMuPDF layout analysis.

Detects figure and table captions by scanning PDF text blocks whose first line
matches a known caption-start pattern (e.g. "Figure 1.", "Table 2:").
Spatial proximity is then used to associate each caption with the nearest
image block (for figure captions) or table bounding box (for table captions)
on the same page.
"""
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import pymupdf

from app.multimodal.models import (
    CAPTION_TYPE_FIGURE,
    CAPTION_TYPE_TABLE,
    CaptionMatch,
    ExtractedCaption,
    ExtractedImage,
    ExtractedTable,
)

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Matches the start of a caption line, e.g.:
#   "Figure 1."  "Figure 1:"  "Fig. 2"  "FIGURE 3 –"
#   "Table 1."   "Table 2:"   "Tab. 3"  "TABLE 4"
_CAPTION_START_RE = re.compile(
    r"^(Figure|Fig\.?|FIG\.?|Table|Tab\.?|TAB\.?)\s*\.?\s*(\d+)\s*[\.:\-–—]?",
    re.IGNORECASE,
)

# Identifies whether the prefix word is a figure or table variant
_FIGURE_PREFIX_RE = re.compile(r"^(Figure|Fig\.?|FIG\.?)", re.IGNORECASE)
_TABLE_PREFIX_RE = re.compile(r"^(Table|Tab\.?|TAB\.?)", re.IGNORECASE)

# Maximum vertical distance (PDF points) within which a caption is considered
# spatially proximate to a visual element.  Captions further away than this
# are still extracted but will not produce a CaptionMatch.
_MAX_PROXIMITY_PTS: float = 72.0  # 1 inch


def _bbox_vertical_distance(
    bbox_a: Tuple[float, float, float, float],
    bbox_b: Tuple[float, float, float, float],
) -> float:
    """Returns the vertical gap between two bounding boxes in PDF points.

    Zero is returned when the boxes overlap vertically.
    """
    # a: (ax0, ay0, ax1, ay1),  b: (bx0, by0, bx1, by1)
    a_top, a_bottom = bbox_a[1], bbox_a[3]
    b_top, b_bottom = bbox_b[1], bbox_b[3]
    # Gap = max(0, max(tops) - min(bottoms)) only when no vertical overlap
    overlap = min(a_bottom, b_bottom) - max(a_top, b_top)
    if overlap >= 0:
        return 0.0  # boxes overlap vertically
    return abs(overlap)


def _parse_caption_block(
    block: Dict[str, Any],
) -> Optional[Tuple[str, str, str]]:
    """Parses a text block and returns (caption_type, label, full_text) or None."""
    lines = block.get("lines", [])
    if not lines:
        return None

    first_line_text = "".join(
        s.get("text", "") for s in lines[0].get("spans", [])
    ).strip()

    m = _CAPTION_START_RE.match(first_line_text)
    if not m:
        return None

    prefix = m.group(1)
    number = m.group(2)

    if _FIGURE_PREFIX_RE.match(prefix):
        cap_type = CAPTION_TYPE_FIGURE
        label = f"Figure {number}"
    else:
        cap_type = CAPTION_TYPE_TABLE
        label = f"Table {number}"

    # Concatenate all lines of the block into the full caption text
    full_text = " ".join(
        "".join(s.get("text", "") for s in ln.get("spans", [])).strip()
        for ln in lines
    ).strip()

    return cap_type, label, full_text


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CaptionExtractionResult:
    """Groups captions and caption-element matches returned by the extractor.

    Attributes:
        captions: All extracted captions from the document.
        matches: Spatial matches between captions and visual elements.
    """

    captions: List[ExtractedCaption] = field(default_factory=list)
    matches: List[CaptionMatch] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


class CaptionExtractor:
    """Extracts figure and table captions from research PDF pages using PyMuPDF.

    Caption detection uses a two-pass approach per page:

    1. **Text scan** – text blocks whose first line matches a caption-start
       pattern are classified as figure or table captions and collected.

    2. **Spatial association** – each caption is paired with the nearest
       visual element of the corresponding type on the same page using
       vertical bounding-box proximity.  Figure captions are matched against
       embedded image blocks; table captions are matched against the bounding
       boxes of any supplied ``ExtractedTable`` objects.

    This extractor does **not** introduce any dependencies beyond PyMuPDF.
    """

    def __init__(self, max_proximity_pts: float = _MAX_PROXIMITY_PTS) -> None:
        """
        Args:
            max_proximity_pts: Maximum vertical gap (PDF points) for a caption
                to be considered spatially associated with an element.
                Defaults to 72 pts (1 inch).
        """
        self.max_proximity_pts = float(max_proximity_pts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        pdf_path: str | Path,
        images: Optional[List[ExtractedImage]] = None,
        tables: Optional[List[ExtractedTable]] = None,
    ) -> CaptionExtractionResult:
        """Extracts captions and spatial matches from the given PDF.

        Args:
            pdf_path: File system path to the PDF document.
            images: Optional list of already-extracted images (from
                ``ImageExtractor``) used for spatial figure-caption matching.
                When omitted the extractor discovers image blocks directly
                from the PDF's layout dictionary.
            tables: Optional list of already-extracted tables (from
                ``TableExtractor``) used for spatial table-caption matching.

        Returns:
            ``CaptionExtractionResult`` with ``captions`` and ``matches``.
            Both lists are empty when the PDF contains no captions.

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

        result = CaptionExtractionResult()
        try:
            # Build per-page index of image bbox lists from supplied ExtractedImage
            # objects, falling back to image blocks from the PDF layout.
            img_bboxes_by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
            img_ids_by_page: Dict[int, List[str]] = {}

            if images:
                for img in images:
                    pg = img.page_number
                    img_bboxes_by_page.setdefault(pg, [])
                    img_ids_by_page.setdefault(pg, [])
                    # ExtractedImage carries no bbox; we rely on layout block below.
                    # Store a placeholder that we'll fill during page iteration.

            # Build per-page index of table bbox + id from supplied ExtractedTable objects.
            tbl_bboxes_by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
            tbl_ids_by_page: Dict[int, List[str]] = {}
            if tables:
                for tbl in tables:
                    if tbl.bbox is None:
                        continue
                    pg = tbl.page_number
                    tbl_bboxes_by_page.setdefault(pg, [])
                    tbl_ids_by_page.setdefault(pg, [])
                    tbl_bboxes_by_page[pg].append(tbl.bbox)
                    tbl_ids_by_page[pg].append(tbl.table_id)

            for page_idx, page in enumerate(doc, start=1):
                try:
                    page_dict = page.get_text("dict")
                except Exception:
                    page_dict = {}

                blocks = page_dict.get("blocks", [])

                # Collect image-block bboxes from the layout (type == 1)
                layout_img_bboxes: List[Tuple[float, float, float, float]] = []
                for blk in blocks:
                    if blk.get("type") == 1:
                        raw_bb = blk.get("bbox")
                        if raw_bb:
                            try:
                                layout_img_bboxes.append(
                                    tuple(float(x) for x in raw_bb)
                                )
                            except Exception:
                                pass

                # Merge supplied image bboxes from ExtractedImage if any.
                # (ExtractedImage has no bbox field; layout blocks are the source.)
                page_img_bboxes = layout_img_bboxes

                # ----------------------------------------------------------
                # Pass 1 – detect caption blocks
                # ----------------------------------------------------------
                cap_idx = 0
                for blk in blocks:
                    if blk.get("type") != 0:
                        continue

                    parsed = _parse_caption_block(blk)
                    if parsed is None:
                        continue

                    cap_type, label, full_text = parsed

                    raw_bb = blk.get("bbox")
                    bbox: Optional[Tuple[float, float, float, float]] = None
                    if raw_bb:
                        try:
                            bbox = tuple(float(x) for x in raw_bb)
                        except Exception:
                            bbox = None

                    caption_id = f"{path.name}#page_{page_idx}_cap_{cap_idx}"
                    cap = ExtractedCaption(
                        caption_id=caption_id,
                        source_filename=path.name,
                        page_number=page_idx,
                        caption_index=cap_idx,
                        caption_type=cap_type,
                        text=full_text,
                        label=label,
                        bbox=bbox,
                    )
                    result.captions.append(cap)
                    cap_idx += 1

            # ----------------------------------------------------------
            # Pass 2 – spatial association (per document)
            # ----------------------------------------------------------
            # Re-iterate the document page by page using the collected captions.
            self._associate_captions(
                result=result,
                doc=doc,
                path=path,
                tables=tables,
                images=images,
            )

        except (FileNotFoundError, ValueError):
            raise
        except Exception as exc:
            raise ValueError(
                f"Failed to extract captions from PDF '{path}': {exc}"
            ) from exc
        finally:
            doc.close()

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _associate_captions(
        self,
        result: CaptionExtractionResult,
        doc: Any,
        path: Path,
        tables: Optional[List[ExtractedTable]],
        images: Optional[List[ExtractedImage]] = None,
    ) -> None:
        """Matches each caption to the nearest visual element using bounding boxes."""
        # Build per-page index of table bboxes+ids
        tbl_bboxes_by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
        tbl_ids_by_page: Dict[int, List[str]] = {}
        if tables:
            for tbl in tables:
                if tbl.bbox is None:
                    continue
                pg = tbl.page_number
                tbl_bboxes_by_page.setdefault(pg, [])
                tbl_ids_by_page.setdefault(pg, [])
                tbl_bboxes_by_page[pg].append(tbl.bbox)
                tbl_ids_by_page[pg].append(tbl.table_id)

        # Build per-page index of image-block bboxes from layout
        img_bboxes_by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
        img_ids_by_page: Dict[int, List[str]] = {}
        supplied_image_ids_by_page: Dict[int, List[str]] = {}
        if images:
            for image in images:
                supplied_image_ids_by_page.setdefault(image.page_number, []).append(
                    image.image_id
                )
        for page_idx, page in enumerate(doc, start=1):
            try:
                page_dict = page.get_text("dict")
            except Exception:
                page_dict = {}
            blocks = page_dict.get("blocks", [])
            img_bboxes: List[Tuple[float, float, float, float]] = []
            for b_idx, blk in enumerate(blocks):
                if blk.get("type") == 1:
                    raw_bb = blk.get("bbox")
                    if raw_bb:
                        try:
                            img_bboxes.append(tuple(float(x) for x in raw_bb))
                            page_image_ids = supplied_image_ids_by_page.get(page_idx, [])
                            image_index = blk.get("number")
                            if (
                                not isinstance(image_index, int)
                                or image_index < 0
                            ):
                                image_index = len(img_bboxes) - 1
                            if image_index < len(page_image_ids):
                                image_id = page_image_ids[image_index]
                            else:
                                image_id = (
                                    f"{path.name}#page_{page_idx}_img_{image_index}"
                                )
                            img_ids_by_page.setdefault(page_idx, []).append(image_id)
                        except Exception:
                            pass
            img_bboxes_by_page[page_idx] = img_bboxes
            img_ids_by_page.setdefault(page_idx, [])

        for cap in result.captions:
            if cap.bbox is None:
                continue

            pg = cap.page_number

            if cap.is_figure_caption:
                bboxes = img_bboxes_by_page.get(pg, [])
                ids = img_ids_by_page.get(pg, [])
            else:
                bboxes = tbl_bboxes_by_page.get(pg, [])
                ids = tbl_ids_by_page.get(pg, [])

            if not bboxes:
                continue

            best_dist = float("inf")
            best_id = None
            for elem_id, elem_bbox in zip(ids, bboxes):
                dist = _bbox_vertical_distance(cap.bbox, elem_bbox)
                if dist < best_dist:
                    best_dist = dist
                    best_id = elem_id

            if best_id is not None and best_dist <= self.max_proximity_pts:
                result.matches.append(
                    CaptionMatch(
                        element_id=best_id,
                        caption_id=cap.caption_id,
                        caption_type=cap.caption_type,
                        proximity_pts=best_dist,
                        source_filename=path.name,
                        page_number=pg,
                    )
                )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def extract_captions(
    pdf_path: str | Path,
    images: Optional[List[ExtractedImage]] = None,
    tables: Optional[List[ExtractedTable]] = None,
    max_proximity_pts: float = _MAX_PROXIMITY_PTS,
) -> CaptionExtractionResult:
    """Convenience function to extract figure/table captions from a PDF.

    Args:
        pdf_path: File system path to the PDF document.
        images: Optional pre-extracted images used for spatial matching.
        tables: Optional pre-extracted tables used for spatial matching.
        max_proximity_pts: Maximum vertical proximity for a spatial match.

    Returns:
        ``CaptionExtractionResult`` with ``captions`` and ``matches`` lists.
    """
    extractor = CaptionExtractor(max_proximity_pts=max_proximity_pts)
    return extractor.extract(pdf_path, images=images, tables=tables)
