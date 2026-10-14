from pathlib import Path
import re
from typing import List, Optional, Tuple
import pymupdf

from app.multimodal.models import ExtractedEquation

# Mathematical Unicode symbols commonly found in scientific formulas
MATH_SYMBOLS = set(
    "∑∫∏√∂∇∈∉⊂⊆∪∩≤≥≠≈≡∝∞±∓×÷⊗⊕→←⇒⇔∀∃αβγδεθλμπστφψωΔΣΩΓ"
)

# Common LaTeX math macros
MATH_MACROS = re.compile(
    r"\\(?:sum|int|prod|frac|sqrt|partial|nabla|alpha|beta|gamma|delta|epsilon|"
    r"theta|lambda|mu|pi|sigma|tau|phi|psi|omega|mathbf|mathit|mathrm|mathcal|"
    r"left|right|begin|end|times|cdot|approx|le|ge|neq|equiv|sim)\b"
)

# Equation labels at the end of lines, e.g. '(1)', '(2.3)', '[1]', 'Eq. (1)'
EQUATION_LABEL_RE = re.compile(
    r"(?:\((?:[A-Za-z0-9]+[.-])?[0-9]+[a-z]?\)|\[(?:[A-Za-z0-9]+[.-])?[0-9]+[a-z]?\]|\bEq(?:uation)?\.?\s*\(?[0-9]+[a-z]?\)?)$"
)

# Core mathematical relations and definition operators
RELATION_OPS = ["=", "≈", "≤", "≥", "≠", "≡", ":=", "∝", "<=", ">=", "==", "!=", "->", "=>"]

# Common prose words that strongly indicate standard English sentences rather than formulas
PROSE_STOPWORDS = {
    "the", "and", "with", "that", "this", "from", "were", "which", "their", "there",
    "have", "been", "about", "other", "these", "could", "would", "should", "between",
    "through", "during", "before", "after", "using", "used", "where", "when", "into", "over",
}

# Prefixes for figures, tables, algorithms, and sections to prevent false positives
NON_EQUATION_PREFIX_RE = re.compile(
    r"^(?:Figure|Fig\.|Table|Tab\.|Algorithm|Alg\.|Section|Sec\.)\s+[0-9]+",
    re.IGNORECASE,
)


def _classify_equation_line(line: str) -> Tuple[bool, Optional[str]]:
    """Determines whether a single line of text represents a mathematical formula.

    Returns:
        A tuple (is_equation, equation_label).
    """
    s = line.strip()
    if not s or len(s) < 3:
        return False, None

    if NON_EQUATION_PREFIX_RE.match(s):
        return False, None

    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", s)]
    prose_count = sum(1 for w in words if w in PROSE_STOPWORDS)
    if prose_count >= 2:
        return False, None

    label_match = EQUATION_LABEL_RE.search(s)
    label = label_match.group(0) if label_match else None

    has_rel = any(op in s for op in RELATION_OPS) or bool(MATH_MACROS.search(s))
    has_math_sym = any(c in MATH_SYMBOLS for c in s) or bool(MATH_MACROS.search(s))

    # Pattern A: Has an equation label and at least one math operator/symbol
    if label and (has_rel or has_math_sym or re.search(r"[\+\-\*/\^_]", s)):
        return True, label

    # Pattern B: Mathematical relation with math symbols or low word count
    if has_rel and (has_math_sym or re.search(r"[\+\-\*/\^_]", s) or len(words) <= 3):
        return True, label

    return False, None


def _union_bbox(
    b1: Optional[Tuple[float, float, float, float]],
    b2: Optional[Tuple[float, float, float, float]],
) -> Optional[Tuple[float, float, float, float]]:
    """Computes the union bounding box of two boxes."""
    if b1 is None:
        return b2
    if b2 is None:
        return b1
    return (
        min(b1[0], b2[0]),
        min(b1[1], b2[1]),
        max(b1[2], b2[2]),
        max(b1[3], b2[3]),
    )


class EquationExtractor:
    """Extracts mathematical equations and formulas from research PDF pages using PyMuPDF."""

    def __init__(self) -> None:
        pass

    def extract(self, pdf_path: str | Path) -> List[ExtractedEquation]:
        """Extracts equations from the given PDF document.

        Preserves exact 1-based page provenance and deterministic equation indexing.

        Args:
            pdf_path: File system path to the PDF document.

        Returns:
            List of ExtractedEquation instances representing formulas found.
            Returns an empty list if the PDF contains no equations.

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

        extracted: List[ExtractedEquation] = []
        try:
            for page_idx, page in enumerate(doc, start=1):
                try:
                    page_dict = page.get_text("dict")
                except Exception:
                    page_dict = {}

                blocks = page_dict.get("blocks", [])
                eq_idx = 0

                for block in blocks:
                    if block.get("type") != 0:
                        continue

                    lines = block.get("lines", [])
                    i = 0
                    while i < len(lines):
                        line = lines[i]
                        line_text = "".join(
                            span.get("text", "") for span in line.get("spans", [])
                        ).strip()

                        is_eq, label = _classify_equation_line(line_text)
                        if not is_eq:
                            i += 1
                            continue

                        # Candidate equation found; check for continuation lines
                        current_text = line_text
                        current_bbox = line.get("bbox")
                        if current_bbox:
                            current_bbox = tuple(current_bbox)

                        # Look ahead for multiline equation continuation within the block
                        j = i + 1
                        while j < len(lines):
                            next_line = lines[j]
                            next_text = "".join(
                                span.get("text", "") for span in next_line.get("spans", [])
                            ).strip()

                            # Check if next line is continuation (starts with operator or ends with label)
                            is_next_eq, next_label = _classify_equation_line(next_text)
                            starts_with_op = bool(
                                re.match(r"^[\+\-\*/=≤≥≈±·,]\s*", next_text)
                            )

                            if starts_with_op or (is_next_eq and not label and next_label):
                                current_text = f"{current_text} {next_text}"
                                next_bbox = next_line.get("bbox")
                                if next_bbox:
                                    current_bbox = _union_bbox(current_bbox, tuple(next_bbox))
                                if not label and next_label:
                                    label = next_label
                                j += 1
                            else:
                                break

                        equation_id = f"{path.name}#page_{page_idx}_eq_{eq_idx}"

                        extracted.append(
                            ExtractedEquation(
                                equation_id=equation_id,
                                source_filename=path.name,
                                page_number=page_idx,
                                equation_index=eq_idx,
                                text=current_text,
                                equation_label=label,
                                bbox=current_bbox,
                                is_display=True,
                            )
                        )
                        eq_idx += 1
                        i = j
        except Exception as exc:
            raise ValueError(
                f"Failed to extract equations from PDF '{path}': {exc}"
            ) from exc
        finally:
            doc.close()

        return extracted


def extract_equations(pdf_path: str | Path) -> List[ExtractedEquation]:
    """Convenience function to extract mathematical formulas and equations from a PDF.

    Args:
        pdf_path: File system path to the PDF document.

    Returns:
        List of ExtractedEquation instances.
    """
    extractor = EquationExtractor()
    return extractor.extract(pdf_path)
