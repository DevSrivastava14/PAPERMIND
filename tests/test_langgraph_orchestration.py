from pathlib import Path
from unittest.mock import Mock, patch

from app.indexing.multimodal_models import MultimodalSearchResult
from app.multimodal.collection import MultimodalDocument
from app.multimodal.representation import ElementType, MultimodalElement
from app.orchestration.langgraph_graph import (
    build_papermind_graph,
    build_retrieval_pipeline,
    ingest_document,
    retrieve_context,
)
from app.reranking.multimodal import MultimodalReranker
from app.retrieval.multimodal_models import (
    MultimodalCitation,
    MultimodalRetrievedContext,
)


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


def test_build_retrieval_pipeline_uses_multimodal_reranker_when_images_present() -> None:
    doc = MultimodalDocument(source_filename="paper.pdf")
    doc.add_element(
        MultimodalElement(
            element_id="paper.pdf#chunk_0",
            element_type=ElementType.TEXT,
            source_filename="paper.pdf",
            page_number=1,
            content="Some text content",
        )
    )
    doc.add_element(
        MultimodalElement(
            element_id="paper.pdf#page_1_img_0",
            element_type=ElementType.IMAGE,
            source_filename="paper.pdf",
            page_number=1,
            content=b"fake-image-bytes",
        )
    )

    with patch("app.orchestration.langgraph_graph.create_evaluation_pipeline") as mock_create:
        mock_create.return_value = Mock()
        result = build_retrieval_pipeline({"document": doc, "top_k": 5})

    assert result["status"] == "indexing_complete"
    assert "reranker" in mock_create.call_args.kwargs
    configured_reranker = mock_create.call_args.kwargs["reranker"]
    assert isinstance(configured_reranker, MultimodalReranker)


def test_build_retrieval_pipeline_uses_text_cross_encoder_when_no_images_present() -> None:
    doc = MultimodalDocument(source_filename="paper.pdf")
    doc.add_element(
        MultimodalElement(
            element_id="paper.pdf#chunk_0",
            element_type=ElementType.TEXT,
            source_filename="paper.pdf",
            page_number=1,
            content="Text-only content",
        )
    )

    with patch("app.orchestration.langgraph_graph.create_evaluation_pipeline") as mock_create:
        mock_create.return_value = Mock()
        result = build_retrieval_pipeline({"document": doc, "top_k": 5})

    assert result["status"] == "indexing_complete"
    assert "reranker" not in mock_create.call_args.kwargs


def test_build_retrieval_pipeline_respects_custom_reranker() -> None:
    doc = MultimodalDocument(source_filename="paper.pdf")
    custom_reranker = Mock()

    with patch("app.orchestration.langgraph_graph.create_evaluation_pipeline") as mock_create:
        mock_create.return_value = Mock()
        result = build_retrieval_pipeline(
            {"document": doc, "top_k": 5, "reranker": custom_reranker}
        )

    assert result["status"] == "indexing_complete"
    assert mock_create.call_args.kwargs.get("reranker") is custom_reranker


def test_graph_retrieval_retains_image_result_for_figure_query() -> None:
    """Proves LangGraph retrieve_context returns visual results when multimodal reranker retains an image."""
    img_element = MultimodalElement(
        element_id="paper.pdf#page_2_img_1",
        element_type=ElementType.IMAGE,
        source_filename="paper.pdf",
        page_number=2,
        content=b"diagram-image",
    )
    img_search_result = MultimodalSearchResult(
        element=img_element,
        score=0.85,
        element_id=img_element.element_id,
        element_type=ElementType.IMAGE,
        source_filename="paper.pdf",
        page_number=2,
    )
    cap_element = MultimodalElement(
        element_id="paper.pdf#page_2_cap_1",
        element_type=ElementType.CAPTION,
        source_filename="paper.pdf",
        page_number=2,
        content="Figure 2: Architecture diagram.",
    )
    cap_search_result = MultimodalSearchResult(
        element=cap_element,
        score=0.95,
        element_id=cap_element.element_id,
        element_type=ElementType.CAPTION,
        source_filename="paper.pdf",
        page_number=2,
    )

    cit_cap = MultimodalCitation(
        citation_label="[1]",
        element_id=cap_element.element_id,
        element_type=ElementType.CAPTION,
        source_filename="paper.pdf",
        page_number=2,
        score=0.95,
    )
    cit_img = MultimodalCitation(
        citation_label="[2]",
        element_id=img_element.element_id,
        element_type=ElementType.IMAGE,
        source_filename="paper.pdf",
        page_number=2,
        score=0.85,
    )
    multimodal_context = MultimodalRetrievedContext(
        formatted_context=(
            "[1] [Caption] Source: paper.pdf (Page 2)\nFigure 2: Architecture diagram.\n\n"
            "[2] [Image] Source: paper.pdf (Page 2)\n[Image: paper.pdf#page_2_img_1]"
        ),
        citations=[cit_cap, cit_img],
        search_results=[cap_search_result, img_search_result],
    )

    pipeline = Mock()
    pipeline.hybrid_retrieve_reranked_and_assemble.return_value = multimodal_context

    doc = MultimodalDocument(
        elements=[cap_element, img_element],
        source_filename="paper.pdf",
    )
    state = {
        "pipeline": pipeline,
        "document": doc,
        "query": "What does Figure 2 show?",
        "top_k": 5,
    }

    result = retrieve_context(state)

    assert result["status"] == "success"
    context = result["context"]
    assert isinstance(context, MultimodalRetrievedContext)
    assert len(context.visual_results) == 1
    assert context.visual_results[0].element_type == ElementType.IMAGE
    assert context.visual_results[0].element_id == "paper.pdf#page_2_img_1"
    image_citations = [c for c in context.citations if c.element_type == ElementType.IMAGE]
    assert len(image_citations) == 1
    assert image_citations[0].citation_label == "[2]"


def test_graph_q6_figure_query_retrieves_image_on_canonical_pdf() -> None:
    """Verifies that the canonical sample paper Q6 figure query retains the image with MultimodalReranker at top_k=5."""
    from app.generation.base import BaseGenerator

    class DummyGenerator(BaseGenerator):
        @property
        def model_name(self) -> str:
            return "dummy-test"

        def generate(self, prompt: str) -> str:
            return "Figure 2 shows Weak AI vs Strong AI [1]."

    result = build_papermind_graph().invoke(
        {
            "pdf_path": "data/sample_paper.pdf",
            "query": "What does Figure 2 show, and how are Weak AI and Strong AI broken down?",
            "top_k": 5,
            "generator": DummyGenerator(),
        }
    )

    assert result["status"] == "success"
    context = result["context"]
    assert isinstance(context, MultimodalRetrievedContext)
    assert len(context.visual_results) >= 1
    assert any("img" in r.element_id for r in context.visual_results)


