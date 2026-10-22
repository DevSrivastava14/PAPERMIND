"""Thin FastAPI adapter for the PaperMind LangGraph workflow."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.orchestration.langgraph_graph import GraphState, build_papermind_graph


class QueryRequest(BaseModel):
    """Input required to run the PaperMind retrieval graph."""

    pdf_path: str = Field(min_length=1)
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0)


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
