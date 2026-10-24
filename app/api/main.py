"""Thin FastAPI adapter for the PaperMind LangGraph workflow."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.orchestration.langgraph_graph import GraphState, build_papermind_graph


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


@app.post("/query", response_model=QueryResponse, response_model_exclude_unset=True)
def query_papermind(request: QueryRequest) -> QueryResponse:
    """Run the existing PaperMind graph for a PDF path and query."""
    state: GraphState = graph.invoke(
        {
            "pdf_path": request.pdf_path,
            "query": request.query,
            "top_k": request.top_k,
        }
    )

    if state.get("status") == "error":
        raise HTTPException(
            status_code=422,
            detail=state.get("error") or "PaperMind graph failed",
        )

    context = state.get("context")
    status = state.get("status")
    if status != "success" or context is None:
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
