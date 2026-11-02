from contextlib import contextmanager

import pytest

from app.observability import langsmith_tracing
from app.observability.langsmith_tracing import (
    research_query_trace,
    safe_trace_inputs,
    safe_trace_output,
    trace_stage,
)


def test_trace_inputs_keep_metadata_without_sensitive_values() -> None:
    safe = safe_trace_inputs(
        {
            "query": "private research question",
            "pdf_path": r"C:\private\paper.pdf",
            "state": {
                "status": "success",
                "document_id": "doc-123",
                "top_k": 5,
                "use_cache": True,
            },
        }
    )

    assert safe["query"] == {"present": True, "length": 25}
    assert safe["pdf_path"] == {"present": True}
    assert safe["state"] == {
        "status": "success",
        "document_id_present": True,
        "top_k": 5,
        "use_cache": True,
        "skip_generation": None,
    }
    assert "private research question" not in repr(safe)
    assert "paper.pdf" not in repr(safe)


def test_trace_outputs_keep_counts_without_answer_or_evidence() -> None:
    safe = safe_trace_output(
        {
            "status": "success",
            "candidate_count": 4,
            "answer": "private generated answer",
            "context": "private evidence",
        }
    )

    assert safe["status"] == "success"
    assert safe["candidate_count"] == 4
    assert "private generated answer" not in repr(safe)
    assert "private evidence" not in repr(safe)


def test_process_trace_has_one_parent_and_nested_stages(monkeypatch) -> None:
    active: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []

    @contextmanager
    def fake_trace(name, **kwargs):
        run = {
            "name": name,
            "parent": active[-1]["name"] if active else None,
            "status": "running",
        }
        runs.append(run)
        active.append(run)
        try:
            yield run
        except Exception:
            run["status"] = "error"
            raise
        else:
            run["status"] = "success"
        finally:
            active.pop()

    monkeypatch.setattr(langsmith_tracing, "trace", fake_trace)

    with research_query_trace("req-1", document_id="doc-1", query_length=4, top_k=5):
        with trace_stage("Hybrid Retrieval", result_count=2):
            pass
        with trace_stage("Answer Validation", citation_count=1):
            pass

    assert [(run["name"], run["parent"]) for run in runs] == [
        ("Research Query", None),
        ("Hybrid Retrieval", "Research Query"),
        ("Answer Validation", "Research Query"),
    ]
    assert all(run["status"] == "success" for run in runs)


def test_stage_error_remains_visible_to_parent(monkeypatch) -> None:
    active: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []

    @contextmanager
    def fake_trace(name, **kwargs):
        run = {"name": name, "status": "running"}
        runs.append(run)
        active.append(run)
        try:
            yield run
        except Exception:
            run["status"] = "error"
            raise
        else:
            run["status"] = "success"
        finally:
            active.pop()

    monkeypatch.setattr(langsmith_tracing, "trace", fake_trace)

    with pytest.raises(RuntimeError, match="provider failed"):
        with research_query_trace("req-2", document_id=None, query_length=3, top_k=5):
            with trace_stage("Grounded Answer Generation"):
                raise RuntimeError("provider failed")

    assert [run["status"] for run in runs] == ["error", "error"]
