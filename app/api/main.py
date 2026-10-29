"""Thin FastAPI adapter for the PaperMind LangGraph workflow."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.orchestration.langgraph_graph import GraphState, build_papermind_graph

logger = logging.getLogger("papermind.api")


def _get_log_level() -> int:
    level_name = os.getenv("PAPERMIND_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper().strip()
    return getattr(logging, level_name, logging.INFO)


def _sanitize_message(message: str) -> str:
    """Sanitizes sensitive information from log messages."""
    # Redact common API key patterns (Groq, OpenAI, bearer tokens)
    clean = re.sub(r"gsk_[A-Za-z0-9_\-]+", "[REDACTED_API_KEY]", message)
    clean = re.sub(r"sk-[A-Za-z0-9_\-]+", "[REDACTED_API_KEY]", clean)
    # Redact full local filesystem paths (Windows drive letters & Unix roots)
    clean = re.sub(r"[A-Za-z]:\\[^ \t\n\r\"']+", "[PATH]", clean)
    clean = re.sub(r"/(?:Users|home|tmp|var|etc)/[^ \t\n\r\"']+", "[PATH]", clean)
    # Redact workspace directory if present
    workspace = str(Path.cwd().resolve())
    if workspace in clean:
        clean = clean.replace(workspace, "[PATH]")
    return clean


def _log_event(level: int, event_data: dict) -> None:
    """Emits structured JSON logs respecting the configured log level."""
    target_level = _get_log_level()
    logger.setLevel(target_level)
    if logger.isEnabledFor(level):
        logger.log(level, json.dumps(event_data))


def get_allowed_document_roots() -> list[Path]:
    """Returns the list of allowed document directory roots."""
    roots = [Path(os.getenv("PAPER_MIND_DOC_ROOT", ".")).resolve()]
    extra_roots = os.getenv("PAPER_MIND_ALLOWED_ROOTS")
    if extra_roots:
        for p in extra_roots.split(os.pathsep):
            if p.strip():
                roots.append(Path(p.strip()).resolve())
    return roots


def resolve_safe_pdf_path(
    pdf_path: str | Path,
    base_dir: Path | None = None,
) -> Path:
    """Canonically resolves a PDF path and verifies it resides within an allowed boundary."""
    raw_path = Path(pdf_path)
    allowed_roots = (
        [base_dir.resolve()] if base_dir is not None else get_allowed_document_roots()
    )
    primary_root = allowed_roots[0]

    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    else:
        resolved = (primary_root / raw_path).resolve()

    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        raise ValueError("Access denied: path escapes allowed document directory")

    return resolved


def generate_document_id(
    pdf_path: str | Path,
    base_dir: Path | None = None,
) -> str:
    """Derives a deterministic document_id from the resolved, validated PDF path.

    Uses SHA-256 over the normalized path relative to the allowed root so the ID
    remains stable across relative, normalized, or absolute representations of the
    same file, without exposing local filesystem directory paths or file contents.
    """
    resolved = resolve_safe_pdf_path(pdf_path, base_dir=base_dir)
    allowed_roots = (
        [base_dir.resolve()] if base_dir is not None else get_allowed_document_roots()
    )
    rel_identity = None
    for root in allowed_roots:
        if resolved.is_relative_to(root):
            rel_identity = resolved.relative_to(root).as_posix()
            break
    if rel_identity is None:
        rel_identity = resolved.name
    digest = hashlib.sha256(rel_identity.encode("utf-8")).hexdigest()[:16]
    return f"doc-{digest}"


class QueryRequest(BaseModel):
    """Input required to run the PaperMind retrieval graph."""

    pdf_path: str = Field(min_length=1)
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0)

    @field_validator("pdf_path")
    @classmethod
    def validate_pdf_extension(cls, value: str) -> str:
        clean = value.strip()
        if not clean.lower().endswith(".pdf"):
            raise ValueError(f"Unsupported file format for '{value}': only .pdf files are supported")
        resolve_safe_pdf_path(clean)
        return clean

    @field_validator("query")
    @classmethod
    def validate_query_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be empty or whitespace-only")
        return value


class CitationResponse(BaseModel):
    """Public representation of a multimodal context citation."""

    citation_label: str
    element_id: str
    element_type: str
    source_filename: str
    page_number: int
    score: float | None = None


class QueryResponse(BaseModel):
    """Grounded retrieval and answer result returned by the API."""

    status: str
    grounded_context: str | None = None
    citations: list[CitationResponse] = Field(default_factory=list)
    error: str | None = None
    answer: str | None = None
    insufficient_evidence: bool | None = None


app = FastAPI(title="PaperMind API")
graph = build_papermind_graph()


@app.middleware("http")
async def request_id_and_timing_middleware(request: Request, call_next):
    """Assigns or propagates request_id and attaches X-Request-ID response header."""
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Emits structured failure log for 422 validation errors with sanitized reasons."""
    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
    sanitized_reason = _sanitize_message("; ".join(str(err.get("msg", "")) for err in exc.errors()))
    query_length = None
    if isinstance(getattr(exc, "body", None), dict) and isinstance(exc.body.get("query"), str):
        query_length = len(exc.body["query"])
    _log_event(
        logging.WARNING,
        {
            "event": "request_failure",
            "request_id": request_id,
            "query_length": query_length,
            "error_category": "validation_error",
            "error_message": sanitized_reason,
        },
    )
    response = await request_validation_exception_handler(request, exc)
    response.headers["X-Request-ID"] = request_id
    return response


def _categorize_error(error_msg: str) -> str:
    """Categorizes graph or pipeline failures into sanitized categories."""
    lower = error_msg.lower()
    if "ingestion failed" in lower:
        return "ingestion_error"
    if "indexing failed" in lower:
        return "indexing_error"
    if "retrieval failed" in lower:
        return "retrieval_error"
    if "answer generation failed" in lower:
        return "generation_error"
    return "graph_error"


@app.post("/query", response_model=QueryResponse, response_model_exclude_unset=True)
def query_papermind(
    request: QueryRequest,
    raw_request: Request,
    response: Response,
) -> QueryResponse:
    """Run the existing PaperMind graph for a PDF path and query with latency and request tracing."""
    start_time = time.perf_counter()
    request_id = getattr(raw_request.state, "request_id", None) or raw_request.headers.get("x-request-id") or str(uuid.uuid4())
    document_id = generate_document_id(request.pdf_path)
    response.headers["X-Request-ID"] = request_id

    # Emit request start event (safe metadata only; never log full query or prompts)
    _log_event(
        logging.INFO,
        {
            "event": "request_start",
            "request_id": request_id,
            "document_id": document_id,
            "query_length": len(request.query),
            "top_k": request.top_k,
        },
    )

    try:
        state: GraphState = graph.invoke(
            {
                "pdf_path": request.pdf_path,
                "query": request.query,
                "top_k": request.top_k,
                "request_id": request_id,
                "document_id": document_id,
            }
        )
    except Exception as exc:
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        sanitized_exc = _sanitize_message(str(exc))
        _log_event(
            logging.ERROR,
            {
                "event": "request_failure",
                "request_id": request_id,
                "document_id": document_id,
                "query_length": len(request.query),
                "error_category": "internal_error",
                "error_message": sanitized_exc,
                "total_latency_ms": round(total_latency_ms, 2),
            },
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        )

    if state.get("status") == "error":
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        raw_error = state.get("error") or "PaperMind graph failed"
        category = _categorize_error(raw_error)
        sanitized_error = _sanitize_message(raw_error)
        _log_event(
            logging.WARNING,
            {
                "event": "request_failure",
                "request_id": request_id,
                "document_id": document_id,
                "query_length": len(request.query),
                "error_category": category,
                "error_message": sanitized_error,
                "total_latency_ms": round(total_latency_ms, 2),
            },
        )
        raise HTTPException(
            status_code=422,
            detail=sanitized_error,
        )

    context = state.get("context")
    status = state.get("status")
    if status != "success" or context is None:
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        _log_event(
            logging.ERROR,
            {
                "event": "request_failure",
                "request_id": request_id,
                "document_id": document_id,
                "query_length": len(request.query),
                "error_category": "retrieval_error",
                "error_message": "PaperMind graph did not produce retrieval context",
                "total_latency_ms": round(total_latency_ms, 2),
            },
        )
        raise HTTPException(
            status_code=500,
            detail="PaperMind graph did not produce retrieval context",
        )

    citations = [
        CitationResponse(
            citation_label=citation.citation_label,
            element_id=citation.element_id,
            element_type=citation.element_type.value,
            source_filename=citation.source_filename,
            page_number=citation.page_number,
            score=citation.score,
        )
        for citation in context.citations
    ]

    top_results = [
        {
            "rank": idx + 1,
            "element_id": c.element_id,
            "element_type": c.element_type,
            "score": round(float(c.score), 4) if isinstance(c.score, (int, float)) else None,
        }
        for idx, c in enumerate(citations)
    ]

    pipeline = state.get("pipeline")
    reranker = getattr(pipeline, "reranker", None) or state.get("reranker")
    reranker_name = type(reranker).__name__ if reranker is not None else None
    retrieval_config = {
        "top_k": request.top_k,
        "reranker": reranker_name,
    }
    candidate_count = getattr(context, "candidate_count", None) or state.get("candidate_count")
    result_count = len(top_results)
    llm_latency_ms = state.get("llm_latency_ms")

    total_latency_ms = (time.perf_counter() - start_time) * 1000.0
    _log_event(
        logging.INFO,
        {
            "event": "request_completion",
            "request_id": request_id,
            "document_id": document_id,
            "query_length": len(request.query),
            "top_k": request.top_k,
            "retrieval_config": retrieval_config,
            "candidate_count": candidate_count,
            "result_count": result_count,
            "top_results": top_results,
            "total_latency_ms": round(total_latency_ms, 2),
            "llm_latency_ms": llm_latency_ms,
        },
    )

    answer = state.get("answer")
    if answer is None and state.get("answer_result") is not None:
        answer = state["answer_result"].answer

    insufficient_evidence = state.get("insufficient_evidence")
    if insufficient_evidence is None and state.get("answer_result") is not None:
        insufficient_evidence = state["answer_result"].insufficient_evidence

    kwargs = {
        "status": status,
        "grounded_context": context.formatted_context,
        "citations": citations,
        "error": None,
    }
    if answer is not None:
        kwargs["answer"] = answer
    if insufficient_evidence is not None:
        kwargs["insufficient_evidence"] = insufficient_evidence

    return QueryResponse(**kwargs)

