"""Small LangGraph orchestration layer for PaperMind retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.evaluation.evaluate_retrieval import build_sample_paper_document
from app.evaluation.runner import create_evaluation_pipeline
from app.multimodal.collection import MultimodalDocument
from app.retrieval.multimodal_models import MultimodalRetrievedContext
from app.retrieval.multimodal_pipeline import MultimodalRetrievalPipeline


GraphStatus = Literal[
    "pending",
    "ingestion_complete",
    "indexing_complete",
    "success",
    "error",
]


class GraphState(TypedDict, total=False):
    """Shared state passed between PaperMind graph nodes."""

    pdf_path: str
    query: str
    document: Optional[MultimodalDocument]
    pipeline: Optional[MultimodalRetrievalPipeline]
    context: Optional[MultimodalRetrievedContext]
    error: Optional[str]
    status: GraphStatus
    top_k: int


def ingest_document(state: GraphState) -> GraphState:
    """Build the multimodal document from the configured PDF."""
    try:
        pdf_path = state["pdf_path"]
        document = build_sample_paper_document(Path(pdf_path))
    except (KeyError, FileNotFoundError, ValueError, OSError) as exc:
        return {"status": "error", "error": f"Ingestion failed: {exc}"}

    return {
        "document": document,
        "status": "ingestion_complete",
        "error": None,
    }


def build_retrieval_pipeline(state: GraphState) -> GraphState:
    """Construct the existing FAISS, BM25, and reranked retrieval pipeline."""
    try:
        document = state["document"]
        if document is None:
            raise ValueError("document is required before pipeline construction")
        pipeline = create_evaluation_pipeline(
            document=document,
            default_top_k=state.get("top_k", 5),
        )
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        return {"status": "error", "error": f"Indexing failed: {exc}"}

    return {
        "pipeline": pipeline,
        "status": "indexing_complete",
        "error": None,
    }


def retrieve_context(state: GraphState) -> GraphState:
    """Run hybrid retrieval, reranking, and multimodal context assembly."""
    try:
        query = state["query"]
        pipeline = state["pipeline"]
        document = state["document"]
        if pipeline is None:
            raise ValueError("pipeline is required before retrieval")
        context = pipeline.hybrid_retrieve_reranked_and_assemble(
            query=query,
            top_k=state.get("top_k", 5),
            document=document,
        )
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        return {"status": "error", "error": f"Retrieval failed: {exc}"}

    return {
        "context": context,
        "status": "success",
        "error": None,
    }


def _after_ingestion(state: GraphState) -> Literal["build_pipeline", "__end__"]:
    return "build_pipeline" if state.get("status") == "ingestion_complete" else END


def _after_indexing(state: GraphState) -> Literal["retrieve", "__end__"]:
    return "retrieve" if state.get("status") == "indexing_complete" else END


def build_papermind_graph():
    """Compile the ingestion -> indexing -> retrieval PaperMind graph."""
    graph = StateGraph(GraphState)
    graph.add_node("ingest", ingest_document)
    graph.add_node("build_pipeline", build_retrieval_pipeline)
    graph.add_node("retrieve", retrieve_context)
    graph.add_edge(START, "ingest")
    graph.add_conditional_edges("ingest", _after_ingestion)
    graph.add_conditional_edges("build_pipeline", _after_indexing)
    graph.add_edge("retrieve", END)
    return graph.compile()
