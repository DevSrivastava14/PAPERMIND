from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Sequence


@dataclass
class CitationValidationResult:
    """Structured result of lightweight citation and provenance validation."""

    is_valid: bool
    cited_labels: List[str]
    available_labels: List[str]
    invalid_labels: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def is_insufficient_evidence(answer: str) -> bool:
    """Check whether the answer text indicates insufficient evidence to answer the query."""
    if not answer or not answer.strip():
        return True
    lowered = answer.lower()
    insufficient_indicators = [
        "cannot answer this question based on the provided",
        "cannot answer based on the provided",
        "insufficient evidence",
        "no relevant evidence",
        "not enough information",
        "provided documents do not provide",
        "provided documents do not mention",
        "provided documents do not contain",
        "provided excerpts do not contain",
        "provided excerpts do not mention",
        "provided evidence does not contain",
        "provided evidence is insufficient",
        "no information is provided in the",
        "does not contain sufficient information",
    ]
    return any(indicator in lowered for indicator in insufficient_indicators)


def validate_answer_citations(
    answer: str,
    citations: Sequence[Any],
    is_insufficient: bool = False,
) -> CitationValidationResult:
    """Lightweight validation of answer citations and provenance structure.

    Args:
        answer: The generated answer text.
        citations: Retrieved citation objects matching the evidence excerpts.
        is_insufficient: True if the answer or retrieval state indicates insufficient evidence.

    Returns:
        CitationValidationResult containing validity status, cited labels,
        available labels, invalid labels, and any errors/warnings.
    """
    cited_markers = re.findall(r"\[\d+\]", answer)
    cited_labels = sorted(list(set(cited_markers)))

    available_labels: List[str] = []
    errors: List[str] = []
    warnings: List[str] = []

    for cit in citations:
        label = getattr(cit, "citation_label", None)
        if label:
            available_labels.append(label)

        # Provenance integrity checks
        source = getattr(cit, "source_filename", None)
        if not source or not str(source).strip():
            errors.append(f"Citation {label} is missing source_filename")

        page = getattr(cit, "page_number", None)
        pages = getattr(cit, "page_numbers", None)
        if page is None and not pages:
            errors.append(f"Citation {label} is missing page provenance")
        elif page is not None and (not isinstance(page, int) or page < 1):
            errors.append(f"Citation {label} has invalid page_number: {page}")
        elif pages is not None and (not isinstance(pages, (list, tuple)) or not pages):
            errors.append(f"Citation {label} has invalid page_numbers")

        elem_id = getattr(cit, "element_id", None)
        if elem_id is None or (isinstance(elem_id, str) and not elem_id.strip()):
            chunk_id = getattr(cit, "chunk_id", None)
            if isinstance(chunk_id, str) and chunk_id.strip():
                elem_id = chunk_id
            else:
                elem_id = None

        if not elem_id or (isinstance(elem_id, str) and not elem_id.strip()):
            errors.append(f"Citation {label} is missing element_id or chunk_id")

    available_set = set(available_labels)
    invalid_labels = [label for label in cited_labels if label not in available_set]
    if invalid_labels:
        errors.append(f"Answer cited markers not found in evidence: {invalid_labels}")

    # If evidence was available and this is not an insufficient-evidence response, check citations
    if not is_insufficient and available_labels and not cited_labels:
        warnings.append("Answer does not include inline citations despite available evidence.")

    is_valid = len(errors) == 0
    return CitationValidationResult(
        is_valid=is_valid,
        cited_labels=cited_labels,
        available_labels=available_labels,
        invalid_labels=invalid_labels,
        errors=errors,
        warnings=warnings,
    )
