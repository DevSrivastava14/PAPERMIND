"""One-shot real smoke test — calls Groq for the Day 25 grounded answer generation flow."""
import sys

import pytest

from app.orchestration.langgraph_graph import build_papermind_graph


def test_real_groq_smoke_test() -> None:
    graph = build_papermind_graph()
    result = graph.invoke(
        {
            "pdf_path": "data/sample_paper.pdf",
            "query": (
                "What techniques are used to improve AI security "
                "and what are their main challenges?"
            ),
            "top_k": 3,
        }
    )

    ctx = result.get("context")
    ar = result.get("answer_result")
    ans = result.get("answer", "")
    insuf = result.get("insufficient_evidence", False)
    cits = ctx.citations if ctx else []

    print("\n=== SMOKE TEST RESULT ===")
    print(f"Status             : {result['status']}")
    print(f"Answer generated   : {bool(ans)}")
    print(f"Insufficient evid. : {insuf}")
    print(f"Citation count     : {len(cits)}")

    if cits:
        print("\nCitations / provenance:")
        for c in cits:
            print(
                f"  {c.citation_label}  element_id={c.element_id}"
                f"  type={c.element_type.value}"
                f"  file={c.source_filename}"
                f"  page={c.page_number}"
                f"  score={c.score:.3f}"
            )

    print(f"\nGenerated answer (first 600 chars):\n{ans[:600]}")

    if ar and ar.validation:
        v = ar.validation
        print(f"\nValidation is_valid    : {v.is_valid}")
        print(f"Cited markers in text  : {v.cited_labels}")
        print(f"Invalid/hallucinated   : {v.invalid_labels}")
        if v.errors:
            print(f"Errors   : {v.errors}")
        if v.warnings:
            print(f"Warnings : {v.warnings}")

    # Assertions
    assert result["status"] == "success"
    assert bool(ans), "Expected a non-empty answer"
    assert ctx is not None, "Expected retrieval context"
    assert len(cits) > 0, "Expected at least one citation"
    # All citations must have valid provenance
    for c in cits:
        assert c.source_filename, f"Missing source_filename on {c.citation_label}"
        assert c.page_number >= 1, f"Invalid page_number on {c.citation_label}"
        assert c.element_id, f"Missing element_id on {c.citation_label}"
    # Validation result must be attached and must not flag hallucinated markers
    assert ar is not None and ar.validation is not None
    assert ar.validation.invalid_labels == [], (
        f"Hallucinated citation markers detected: {ar.validation.invalid_labels}"
    )
