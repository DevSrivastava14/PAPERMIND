"""Thin FastAPI adapter for the PaperMind LangGraph workflow."""

from __future__ import annotations

import hashlib
import base64
import json
import logging
import os
from pathlib import Path
import re
import time
import uuid

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_project_environment() -> None:
    """Load project configuration before importing application modules."""
    load_dotenv(_PROJECT_ROOT / ".env")


_load_project_environment()

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from app.multimodal.representation import ElementType, MultimodalElement
from app.orchestration.langgraph_graph import GraphState, build_papermind_graph
from app.evaluation.viewer import list_artifacts, load_artifact
from app.documents.manager import (
    get_upload_root,
    get_document,
    list_documents,
    save_uploaded_pdf,
    update_document_status,
)
from app.generation.answer_generator import build_answer_generator
from app.generation.groq_generator import GroqGenerator
from app.generation.prompt_builder import build_grounded_prompt
from app.generation.validation import is_insufficient_evidence, validate_answer_citations
from app.observability.langsmith_tracing import research_query_trace, trace_stage

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

    pdf_path: str | None = Field(default=None, min_length=1)
    document_id: str | None = Field(default=None, min_length=1)
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0)

    @field_validator("pdf_path")
    @classmethod
    def validate_pdf_extension(cls, value: str) -> str:
        clean = value.strip()
        if not clean.lower().endswith(".pdf"):
            raise ValueError(f"Unsupported file format for '{value}': only .pdf files are supported")
        return clean

    @model_validator(mode="after")
    def validate_document_selector(self) -> "QueryRequest":
        if not self.pdf_path and not self.document_id:
            raise ValueError("pdf_path or document_id is required")
        if self.pdf_path and self.document_id:
            raise ValueError("Provide either pdf_path or document_id, not both")
        if self.pdf_path:
            resolved = resolve_safe_pdf_path(self.pdf_path)
            if resolved.is_relative_to(get_upload_root().resolve()):
                raise ValueError(
                    "Uploaded documents must be queried with document_id"
                )
        return self

    @field_validator("query")
    @classmethod
    def validate_query_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be empty or whitespace-only")
        return value


class RetrievalDiagnosticsResponse(BaseModel):
    """Optional retrieval-stage values preserved for one evidence item."""

    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    rrf_rank: int | None = None
    rrf_dense_contribution: float | None = None
    rrf_bm25_contribution: float | None = None
    cross_encoder_score: float | None = None
    clip_score: float | None = None
    final_score: float | None = None
    final_rank: int | None = None


class CitationResponse(BaseModel):
    """Public representation of a multimodal context citation."""

    citation_label: str
    element_id: str
    element_type: str
    source_filename: str
    page_number: int
    score: float | None = None
    image_data_url: str | None = None
    caption: str | None = None
    related_element_ids: list[str] | None = None
    associated_text: str | None = None
    diagnostics: RetrievalDiagnosticsResponse | None = None


class RetrievalRunDiagnosticsResponse(BaseModel):
    """Optional retrieval-run values available to the technical drawer."""

    candidate_count: int | None = None
    result_count: int | None = None
    reranker: str | None = None
    retrieval_latency_ms: float | None = None


class QueryResponse(BaseModel):
    """Grounded retrieval and answer result returned by the API."""

    status: str
    grounded_context: str | None = None
    citations: list[CitationResponse] = Field(default_factory=list)
    error: str | None = None
    answer: str | None = None
    insufficient_evidence: bool | None = None
    retrieval_diagnostics: RetrievalRunDiagnosticsResponse | None = None


app = FastAPI(title="PaperMind API")
graph = build_papermind_graph()

# ── Static files & root route ──────────────────────────────────────────────
_STATIC_DIR = Path(__file__).parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    """Serve the PaperMind frontend."""
    return FileResponse(_STATIC_DIR / "index.html")



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


def _is_document_processing_failure(error_msg: str) -> bool:
    return error_msg.lower().startswith(
        ("ingestion failed", "indexing failed", "retrieval failed")
    )


def _image_data_url(element: MultimodalElement) -> str | None:
    """Returns an inline image payload without exposing its filesystem path."""
    image_data = element.image_data
    if not image_data:
        return None
    image_format = str(element.metadata.get("image_format") or "png").lower()
    if image_format not in {"png", "jpeg", "jpg", "gif", "webp", "bmp", "tiff"}:
        image_format = "png"
    if image_format == "jpg":
        image_format = "jpeg"
    encoded = base64.b64encode(image_data).decode("ascii")
    return f"data:image/{image_format};base64,{encoded}"


def _related_evidence(
    element: MultimodalElement,
    document: object | None,
) -> tuple[str | None, str | None]:
    """Resolves only existing related caption/text elements from the document."""
    if document is None:
        return None, None
    get_by_id = getattr(document, "get_by_id", None)
    if not callable(get_by_id):
        return None, None

    caption = None
    associated_text: list[str] = []
    for related_id in element.related_element_ids:
        related = get_by_id(related_id)
        if related is None or not related.text.strip():
            continue
        if related.element_type == ElementType.CAPTION and caption is None:
            caption = related.text.strip()
        elif related.element_type != ElementType.IMAGE:
            associated_text.append(related.text.strip())
    return caption, "\n\n".join(associated_text) if associated_text else None


def _citation_metadata(
    citation: object,
    context: object,
    document: object | None,
) -> dict[str, object]:
    """Builds optional multimodal metadata from the original retrieved element."""
    search_results = getattr(context, "search_results", None) or []
    result = next(
        (
            candidate
            for candidate in search_results
            if getattr(candidate, "element_id", None) == getattr(citation, "element_id", None)
        ),
        None,
    )
    element = getattr(result, "element", None)
    if not isinstance(element, MultimodalElement):
        return {}

    metadata: dict[str, object] = {}
    if element.related_element_ids:
        metadata["related_element_ids"] = list(element.related_element_ids)
    if element.element_type != ElementType.IMAGE:
        return metadata


    caption, associated_text = _related_evidence(element, document)
    image_url = _image_data_url(element)
    if image_url is not None:
        metadata["image_data_url"] = image_url
    if caption is not None:
        metadata["caption"] = caption
    if associated_text is not None:
        metadata["associated_text"] = associated_text
    return metadata


def _result_diagnostics(
    result: object | None,
    final_rank: int,
) -> RetrievalDiagnosticsResponse | None:
    if result is None:
        return None
    names = (
        "dense_score",
        "dense_rank",
        "bm25_score",
        "bm25_rank",
        "rrf_score",
        "rrf_rank",
        "rrf_dense_contribution",
        "rrf_bm25_contribution",
        "cross_encoder_score",
        "clip_score",
        "final_score",
    )
    values = {name: getattr(result, name, None) for name in names}
    values["final_rank"] = getattr(result, "final_rank", None) or final_rank
    values = {name: value for name, value in values.items() if value is not None}
    if not values:
        return None
    return RetrievalDiagnosticsResponse(**values)


def _retrieval_response_fields(
    state: GraphState,
    request: QueryRequest,
    document_id: str,
) -> dict[str, object]:
    context = state["context"]
    document = state.get("document")
    search_results = getattr(context, "search_results", None) or []
    citations: list[CitationResponse] = []
    for final_rank, citation in enumerate(context.citations, start=1):
        result = next(
            (
                candidate
                for candidate in search_results
                if getattr(candidate, "element_id", None) == citation.element_id
            ),
            None,
        )
        citation_kwargs = {
            "citation_label": citation.citation_label,
            "element_id": citation.element_id,
            "element_type": citation.element_type.value,
            "source_filename": citation.source_filename,
            "page_number": citation.page_number,
            "score": citation.score,
            **_citation_metadata(citation, context, document),
        }
        diagnostics = _result_diagnostics(result, final_rank)
        if diagnostics is not None:
            citation_kwargs["diagnostics"] = diagnostics
        citations.append(CitationResponse(**citation_kwargs))

    pipeline = state.get("pipeline")
    reranker = getattr(pipeline, "reranker", None) or state.get("reranker")
    candidate_count = getattr(context, "candidate_count", None) or state.get("candidate_count")
    result_count = len(citations)
    fields: dict[str, object] = {
        "grounded_context": context.formatted_context,
        "citations": citations,
    }
    if any(value is not None for value in (candidate_count, reranker, state.get("retrieval_latency_ms"))):
        fields["retrieval_diagnostics"] = RetrievalRunDiagnosticsResponse(
            candidate_count=candidate_count,
            result_count=result_count,
            reranker=type(reranker).__name__ if reranker is not None else None,
            retrieval_latency_ms=state.get("retrieval_latency_ms"),
        )
    return fields


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight health check endpoint returning API health status."""
    return {"status": "healthy"}


@app.get("/documents")
def documents() -> dict[str, list[dict[str, object]]]:
    """List application-managed uploaded documents."""
    try:
        return {"documents": list_documents()}
    except OSError:
        raise HTTPException(status_code=500, detail="Uploaded documents are unavailable")


@app.post("/documents", status_code=201)
def upload_document(file: UploadFile = File(...)) -> dict[str, object]:
    """Validate and persist one PDF without starting document ingestion."""
    filename = file.filename or ""
    try:
        return save_uploaded_pdf(filename, file.file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError:
        raise HTTPException(status_code=500, detail="The uploaded document could not be stored")
    finally:
        file.file.close()


@app.get("/evaluation/results")
def evaluation_results(
    artifact_id: str | None = Query(default=None, min_length=1),
) -> dict:
    """Lists or reads one fixed, persisted evaluation artifact."""
    if artifact_id is None:
        return {"artifacts": list_artifacts()}
    try:
        return {"artifact": load_artifact(artifact_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown evaluation artifact")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Evaluation artifact is unavailable")
    except (OSError, ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="Saved evaluation artifact is invalid")


@app.post("/query", response_model=QueryResponse, response_model_exclude_unset=True)
def query_papermind(
    request: QueryRequest,
    raw_request: Request,
    response: Response,
) -> QueryResponse:
    request_id = getattr(raw_request.state, "request_id", None) or raw_request.headers.get("x-request-id") or str(uuid.uuid4())
    with research_query_trace(
        request_id,
        document_id=request.document_id,
        query_length=len(request.query),
        top_k=request.top_k,
    ):
        return _query_papermind_impl(request, raw_request, response)


def _query_papermind_impl(
    request: QueryRequest,
    raw_request: Request,
    response: Response,
) -> QueryResponse:
    """Run the existing PaperMind graph for a PDF path and query with latency and request tracing."""
    start_time = time.perf_counter()
    request_id = getattr(raw_request.state, "request_id", None) or raw_request.headers.get("x-request-id") or str(uuid.uuid4())
    if request.document_id:
        uploaded_document = get_document(request.document_id)
        if uploaded_document is None:
            raise HTTPException(status_code=404, detail="Unknown document")
        source_pdf_path = Path(str(uploaded_document["path"]))
        document_id = request.document_id
    else:
        source_pdf_path = request.pdf_path
        document_id = generate_document_id(request.pdf_path or "")

    if request.document_id:
        try:
            update_document_status(request.document_id, "processing")
        except (KeyError, OSError):
            raise HTTPException(status_code=404, detail="Unknown document")
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
                "pdf_path": str(source_pdf_path),
                "query": request.query,
                "top_k": request.top_k,
                "request_id": request_id,
                "document_id": document_id,
                "use_cache": True,
            }
        )
    except Exception as exc:
        if request.document_id:
            try:
                update_document_status(request.document_id, "failed")
            except (KeyError, OSError):
                pass
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
        raw_error = state.get("error") or "PaperMind graph failed"
        if request.document_id and _is_document_processing_failure(raw_error):
            try:
                update_document_status(request.document_id, "failed")
            except (KeyError, OSError):
                pass
        elif request.document_id:
            try:
                update_document_status(request.document_id, "ready")
            except (KeyError, OSError):
                pass
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
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

    if request.document_id:
        try:
            update_document_status(request.document_id, "ready")
        except (KeyError, OSError):
            raise HTTPException(status_code=500, detail="Document state could not be updated")

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

    document = state.get("document")
    search_results = getattr(context, "search_results", None) or []
    citations = []
    for final_rank, citation in enumerate(context.citations, start=1):
        result = next(
            (
                candidate
                for candidate in search_results
                if getattr(candidate, "element_id", None) == citation.element_id
            ),
            None,
        )
        citation_kwargs = {
            "citation_label": citation.citation_label,
            "element_id": citation.element_id,
            "element_type": citation.element_type.value,
            "source_filename": citation.source_filename,
            "page_number": citation.page_number,
            "score": citation.score,
            **_citation_metadata(citation, context, document),
        }
        diagnostics = _result_diagnostics(result, final_rank)
        if diagnostics is not None:
            citation_kwargs["diagnostics"] = diagnostics
        citations.append(CitationResponse(**citation_kwargs))

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
    if any(
        value is not None
        for value in (candidate_count, reranker_name, state.get("retrieval_latency_ms"))
    ):
        run_values = {
            "candidate_count": candidate_count,
            "result_count": result_count,
            "reranker": reranker_name,
            "retrieval_latency_ms": state.get("retrieval_latency_ms"),
        }
        kwargs["retrieval_diagnostics"] = RetrievalRunDiagnosticsResponse(
            **{name: value for name, value in run_values.items() if value is not None}
        )
    if answer is not None:
        kwargs["answer"] = answer
    if insufficient_evidence is not None:
        kwargs["insufficient_evidence"] = insufficient_evidence

    return QueryResponse(**kwargs)


def _sse_event(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


def _stream_query_events(
    request: QueryRequest,
    request_id: str,
    source_pdf_path: Path,
    document_id: str,
):
    with research_query_trace(
        request_id,
        document_id=document_id,
        query_length=len(request.query),
        top_k=request.top_k,
    ):
        yield from _stream_query_events_impl(
            request,
            request_id,
            source_pdf_path,
            document_id,
        )


def _stream_query_events_impl(
    request: QueryRequest,
    request_id: str,
    source_pdf_path: Path,
    document_id: str,
):
    processing_succeeded = False
    try:
        if request.document_id:
            update_document_status(request.document_id, "processing")
        state: GraphState = graph.invoke(
            {
                "pdf_path": str(source_pdf_path),
                "query": request.query,
                "top_k": request.top_k,
                "request_id": request_id,
                "document_id": document_id,
                "use_cache": True,
                "skip_generation": True,
            }
        )
        if state.get("status") != "success" or state.get("context") is None:
            raw_error = state.get("error") or "PaperMind graph failed"
            if request.document_id and _is_document_processing_failure(raw_error):
                update_document_status(request.document_id, "failed")
            yield _sse_event("error", {"error": _sanitize_message(raw_error)})
            return

        processing_succeeded = True
        fields = _retrieval_response_fields(state, request, document_id)
        retrieval_payload = {
            "request_id": request_id,
            "grounded_context": fields["grounded_context"],
            "citations": [
                citation.model_dump(exclude_none=True)
                for citation in fields["citations"]
            ],
        }
        diagnostics = fields.get("retrieval_diagnostics")
        if diagnostics is not None:
            retrieval_payload["retrieval_diagnostics"] = diagnostics.model_dump(exclude_none=True)
        yield _sse_event("retrieval", retrieval_payload)

        context = state["context"]
        generator = GroqGenerator()
        answer_generator = build_answer_generator(generator)
        if not context.citations or not str(context.formatted_context).strip():
            result = answer_generator.generate_from_context(
                query=request.query, context=context
            )
            yield _sse_event(
                "complete",
                {
                    "answer": result.answer,
                    "insufficient_evidence": result.insufficient_evidence,
                    "citations": retrieval_payload["citations"],
                    "retrieval_diagnostics": retrieval_payload.get("retrieval_diagnostics"),
                },
            )
            if request.document_id:
                update_document_status(request.document_id, "ready")
            return

        with trace_stage("Grounded Answer Generation", model=generator.model_name):
            prompt = build_grounded_prompt(query=request.query, context=context)
            chunks: list[str] = []
            try:
                for chunk in generator.generate_stream(prompt):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    yield _sse_event("token", {"text": chunk})
            except GeneratorExit:
                return

        answer = "".join(chunks).strip()
        insufficient = is_insufficient_evidence(answer)
        with trace_stage(
            "Answer Validation",
            insufficient_evidence=insufficient,
            citation_count=len(context.citations),
        ):
            validation = validate_answer_citations(
                answer=answer,
                citations=context.citations,
                is_insufficient=insufficient,
            )
        yield _sse_event(
            "complete",
            {
                "answer": answer,
                "insufficient_evidence": insufficient,
                "citation_validation": validation.__dict__,
                "citations": retrieval_payload["citations"],
                "retrieval_diagnostics": retrieval_payload.get("retrieval_diagnostics"),
            },
        )
        if request.document_id:
            update_document_status(request.document_id, "ready")
    except Exception as exc:
        if request.document_id:
            try:
                update_document_status(
                    request.document_id,
                    "ready" if processing_succeeded else "failed",
                )
            except (KeyError, OSError):
                pass
        yield _sse_event("error", {"error": _sanitize_message(str(exc))})


@app.post("/query/stream")
def stream_query_papermind(
    request: QueryRequest,
    raw_request: Request,
    response: Response,
) -> StreamingResponse:
    """Stream retrieval metadata and provider-generated answer chunks over SSE."""
    request_id = (
        getattr(raw_request.state, "request_id", None)
        or raw_request.headers.get("x-request-id")
        or str(uuid.uuid4())
    )
    if request.document_id:
        uploaded_document = get_document(request.document_id)
        if uploaded_document is None:
            raise HTTPException(status_code=404, detail="Unknown document")
        source_pdf_path = Path(str(uploaded_document["path"]))
        document_id = request.document_id
    else:
        source_pdf_path = Path(request.pdf_path or "")
        document_id = generate_document_id(request.pdf_path or "")

    response.headers["X-Request-ID"] = request_id
    return StreamingResponse(
        _stream_query_events(request, request_id, source_pdf_path, document_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Request-ID": request_id},
    )
