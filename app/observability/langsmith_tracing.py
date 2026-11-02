"""Privacy-preserving, process-level LangSmith tracing helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from langsmith import trace


def _safe_value(key: str, value: Any) -> Any:
    lowered = key.lower()
    if lowered in {"query", "prompt", "answer", "formatted_context", "text", "content"}:
        if isinstance(value, str):
            return {"present": bool(value), "length": len(value)}
        return {"present": value is not None}
    if lowered in {"pdf_path", "path", "source_pdf_path"}:
        return {"present": value is not None}
    if lowered in {"self", "generator", "document", "pipeline", "context", "reranker"}:
        return type(value).__name__ if value is not None else None
    if isinstance(value, (list, tuple, set)):
        return {"count": len(value)}
    if isinstance(value, dict):
        return {"keys": sorted(str(item) for item in value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


def safe_trace_inputs(inputs: Any) -> dict[str, Any]:
    """Keep only safe operation metadata in traced inputs."""
    if not isinstance(inputs, dict):
        return {"input_type": type(inputs).__name__}
    safe = {str(key): _safe_value(str(key), value) for key, value in inputs.items()}
    state = inputs.get("state")
    if isinstance(state, dict):
        safe["state"] = {
            "status": state.get("status"),
            "document_id_present": bool(state.get("document_id")),
            "top_k": state.get("top_k"),
            "use_cache": state.get("use_cache"),
            "skip_generation": state.get("skip_generation"),
        }
    return safe


def safe_trace_output(output: Any) -> dict[str, Any]:
    """Summarize traced outputs without serializing evidence or answer text."""
    if isinstance(output, dict):
        result: dict[str, Any] = {}
        for key in ("status", "candidate_count", "retrieval_latency_ms", "llm_latency_ms"):
            if key in output:
                result[key] = output[key]
        for key in ("document", "pipeline", "context", "answer_result", "answer"):
            if key in output:
                value = output[key]
                result[f"{key}_type"] = type(value).__name__ if value is not None else None
        return result
    if isinstance(output, (list, tuple)):
        return {"count": len(output)}
    if isinstance(output, str):
        return {"present": bool(output), "length": len(output)}
    return {"output_type": type(output).__name__}


@contextmanager
def research_query_trace(
    request_id: str,
    *,
    document_id: str | None,
    query_length: int,
    top_k: int,
) -> Iterator[Any]:
    """Create one safe parent trace for a complete research request."""
    with trace(
        "Research Query",
        run_type="chain",
        inputs={
            "request_id": request_id,
            "document_id_present": bool(document_id),
            "query_length": query_length,
            "top_k": top_k,
        },
        metadata={"request_id": request_id, "document_id_present": bool(document_id)},
    ) as run:
        yield run


@contextmanager
def trace_stage(name: str, **metadata: Any) -> Iterator[Any]:
    """Create one concise child process span under the active request trace."""
    with trace(
        name,
        run_type="chain",
        inputs={"stage": name},
        metadata=metadata or None,
    ) as run:
        yield run


def reduce_stream_chunks(chunks: Iterator[str]) -> dict[str, int]:
    """Reduce provider chunks to safe stream metadata."""
    items = list(chunks)
    return {"chunk_count": len(items), "text_length": sum(len(item) for item in items)}
