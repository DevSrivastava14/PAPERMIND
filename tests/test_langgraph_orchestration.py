from pathlib import Path
from unittest.mock import Mock, patch

from app.multimodal.collection import MultimodalDocument
from app.orchestration.langgraph_graph import (
    build_papermind_graph,
    ingest_document,
)
from app.retrieval.multimodal_models import MultimodalRetrievedContext


def test_graph_runs_ingestion_indexing_and_reranked_retrieval() -> None:
    document = MultimodalDocument(source_filename="paper.pdf")
    pipeline = Mock()
    context = Mock()
    pipeline.hybrid_retrieve_reranked_and_assemble.return_value = context

    with (
        patch(
            "app.orchestration.langgraph_graph.build_sample_paper_document",
            return_value=document,
        ),
        patch(
            "app.orchestration.langgraph_graph.create_evaluation_pipeline",
            return_value=pipeline,
        ) as build_pipeline,
    ):
        result = build_papermind_graph().invoke(
            {"pdf_path": "paper.pdf", "query": "What is the result?", "top_k": 3}
        )

    assert result["status"] == "success"
    assert result["document"] is document
    assert result["pipeline"] is pipeline
    assert result["context"] is context
    build_pipeline.assert_called_once_with(document=document, default_top_k=3)
    pipeline.hybrid_retrieve_reranked_and_assemble.assert_called_once_with(
        query="What is the result?", top_k=3, document=document
    )


def test_ingestion_failure_stops_graph() -> None:
    with patch(
        "app.orchestration.langgraph_graph.build_sample_paper_document",
        side_effect=FileNotFoundError("missing.pdf"),
    ), patch(
        "app.orchestration.langgraph_graph.create_evaluation_pipeline"
    ) as build_pipeline:
        result = build_papermind_graph().invoke(
            {"pdf_path": "missing.pdf", "query": "query"}
        )

    assert result["status"] == "error"
    assert "Ingestion failed: missing.pdf" in result["error"]
    build_pipeline.assert_not_called()


def test_ingest_document_uses_pdf_path() -> None:
    document = MultimodalDocument(source_filename="paper.pdf")
    with patch(
        "app.orchestration.langgraph_graph.build_sample_paper_document",
        return_value=document,
    ) as build_document:
        result = ingest_document({"pdf_path": "paper.pdf"})

    assert result["document"] is document
    build_document.assert_called_once_with(Path("paper.pdf"))


def test_indexing_failure_stops_before_retrieval() -> None:
    document = MultimodalDocument(source_filename="paper.pdf")
    with patch(
        "app.orchestration.langgraph_graph.build_sample_paper_document",
        return_value=document,
    ), patch(
        "app.orchestration.langgraph_graph.create_evaluation_pipeline",
        side_effect=RuntimeError("index unavailable"),
    ), patch(
        "app.orchestration.langgraph_graph.retrieve_context"
    ) as retrieve:
        result = build_papermind_graph().invoke(
            {"pdf_path": "paper.pdf", "query": "query"}
        )

    assert result["status"] == "error"
    assert result["error"] == "Indexing failed: index unavailable"
    retrieve.assert_not_called()


def test_retrieval_failure_is_reported() -> None:
    document = MultimodalDocument(source_filename="paper.pdf")
    pipeline = Mock()
    pipeline.hybrid_retrieve_reranked_and_assemble.side_effect = RuntimeError(
        "retrieval unavailable"
    )

    with (
        patch(
            "app.orchestration.langgraph_graph.build_sample_paper_document",
            return_value=document,
        ),
        patch(
            "app.orchestration.langgraph_graph.create_evaluation_pipeline",
            return_value=pipeline,
        ),
    ):
        result = build_papermind_graph().invoke(
            {"pdf_path": "paper.pdf", "query": "query"}
        )

    assert result["status"] == "error"
    assert result["error"] == "Retrieval failed: retrieval unavailable"


def test_graph_compiles() -> None:
    graph = build_papermind_graph()

    assert graph is not None


def test_graph_runs_canonical_pdf_end_to_end() -> None:
    result = build_papermind_graph().invoke(
        {
            "pdf_path": "data/sample_paper.pdf",
            "query": "What is the difference between AI for Security and Security for AI?",
            "top_k": 2,
        }
    )

    context = result["context"]
    assert result["status"] == "success"
    assert result["error"] is None
    assert isinstance(context, MultimodalRetrievedContext)
    assert context.formatted_context
    assert context.citations
    assert len(context.citations) == len(context.search_results)
