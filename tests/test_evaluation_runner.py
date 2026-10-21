"""Targeted unit tests for the retrieval evaluation runner."""

from typing import List, Optional
import numpy as np
import pytest

from app.embeddings.base import BaseEmbedder
from app.evaluation import (
    ALL_RETRIEVAL_CONFIGURATIONS,
    QuestionRetrievalRun,
    RetrievalConfiguration,
    RetrievalEvaluationRunner,
    create_evaluation_pipeline,
    get_retrieval_evaluation_dataset,
)
from app.indexing import (
    BM25Index,
    MultimodalFAISSIndex,
    MultimodalSearchResult,
    build_bm25_index,
    build_multimodal_index,
)
from app.multimodal import (
    ElementType,
    MultimodalDocument,
    MultimodalElement,
    MultimodalEmbedder,
)
from app.multimodal.embedding import BaseVisualEmbedder
from app.reranking.base import BaseReranker
from app.reranking.cross_encoder import CrossEncoderReranker
from app.retrieval.multimodal_pipeline import (
    MultimodalRetrievalPipeline,
    build_multimodal_pipeline,
)


# ===========================================================================
# Deterministic Test Fixtures
# ===========================================================================


class MockTextEmbedder(BaseEmbedder):
    """Deterministic 384-d embedder for fast unit tests."""

    @property
    def dimension(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return "mock-384"

    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        vectors = []
        for text in texts:
            vec = np.zeros(384, dtype=np.float32)
            lower = text.lower()
            if "security" in lower:
                vec[0] = 1.0
            if "poisoning" in lower:
                vec[1] = 1.0
            if "model" in lower:
                vec[2] = 1.0
            norm = np.linalg.norm(vec)
            vectors.append(vec / (norm if norm > 0 else 1.0))
        return np.vstack(vectors)


class MockVisualEmbedder(BaseVisualEmbedder):
    """Deterministic 512-d visual embedder."""

    @property
    def dimension(self) -> int:
        return 512

    @property
    def model_name(self) -> str:
        return "mock-visual-512"

    def embed_images(self, images, batch_size=None) -> np.ndarray:
        vecs = np.zeros((len(images), 512), dtype=np.float32)
        vecs[:, 0] = 1.0
        return vecs


class MockCrossEncoderModel:
    """Deterministic cross-encoder scoring model."""

    def __init__(self, score_map=None, default_score=0.1):
        self.score_map = score_map or {}
        self.default_score = default_score

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            matched = None
            for k, v in self.score_map.items():
                if k.lower() in text.lower():
                    matched = v
                    break
            scores.append(float(matched if matched is not None else self.default_score))
        return scores


@pytest.fixture
def eval_test_document() -> MultimodalDocument:
    """Document containing elements corresponding to evaluation topics."""
    el1 = MultimodalElement(
        element_id="txt-sec-1",
        element_type=ElementType.TEXT,
        source_filename="sample_paper.pdf",
        page_number=2,
        content="‘AI for Security’ means how can we use AI technologies to solve security problems.",
        metadata={"section": "II.B"},
    )
    el2 = MultimodalElement(
        element_id="txt-sec-2",
        element_type=ElementType.TEXT,
        source_filename="sample_paper.pdf",
        page_number=3,
        content="‘Security for AI’ means designing, implementing security measures for AI technologies.",
        metadata={"section": "II.C"},
    )
    el3 = MultimodalElement(
        element_id="txt-poison-1",
        element_type=ElementType.TEXT,
        source_filename="sample_paper.pdf",
        page_number=4,
        content="Data poisoning attacks insert incorrectly labeled data sets into training storage.",
        metadata={"section": "II.C.3"},
    )
    el4 = MultimodalElement(
        element_id="txt-poison-2",
        element_type=ElementType.TEXT,
        source_filename="sample_paper.pdf",
        page_number=4,
        content="Model poisoning attacks target AI models in a development or testing environment.",
        metadata={"section": "II.C.3"},
    )
    return MultimodalDocument(
        elements=[el1, el2, el3, el4],
        source_filename="sample_paper.pdf",
    )


@pytest.fixture
def eval_pipeline(eval_test_document: MultimodalDocument) -> MultimodalRetrievalPipeline:
    """Instantiates a pipeline with FAISS, BM25, and Mock CrossEncoder."""
    fake_model = MockCrossEncoderModel(
        score_map={
            "AI for Security": 0.95,
            "Security for AI": 0.90,
            "poisoning": 0.85,
        }
    )
    return create_evaluation_pipeline(
        document=eval_test_document,
        text_embedder=MockTextEmbedder(),
        visual_embedder=MockVisualEmbedder(),
        reranker=CrossEncoderReranker(model=fake_model),
        default_top_k=3,
    )


# ===========================================================================
# Test Cases
# ===========================================================================


def test_runner_initialization_and_validation(eval_pipeline: MultimodalRetrievalPipeline):
    """Verify runner constructor argument validation."""
    runner = RetrievalEvaluationRunner(pipeline=eval_pipeline, default_top_k=3)
    assert runner.pipeline is eval_pipeline
    assert runner.default_top_k == 3
    assert runner.configurations == ALL_RETRIEVAL_CONFIGURATIONS

    with pytest.raises(TypeError, match="pipeline must be an instance"):
        RetrievalEvaluationRunner(pipeline="not_a_pipeline")  # type: ignore

    with pytest.raises(ValueError, match="default_top_k must be a positive integer"):
        RetrievalEvaluationRunner(pipeline=eval_pipeline, default_top_k=0)

    with pytest.raises(ValueError, match="default_rrf_k must be a positive integer"):
        RetrievalEvaluationRunner(pipeline=eval_pipeline, default_rrf_k=-5)


def test_runner_executes_all_four_configurations_for_single_question(
    eval_pipeline: MultimodalRetrievalPipeline,
):
    """Verify the runner takes an EvaluationQuestion and executes all four configurations."""
    dataset = get_retrieval_evaluation_dataset()
    q1 = dataset["Q1"]

    runner = RetrievalEvaluationRunner(pipeline=eval_pipeline, default_top_k=2)
    run = runner.run_question(q1)

    assert isinstance(run, QuestionRetrievalRun)
    assert run.question.id == "Q1"

    # All 4 configurations executed
    assert set(run.results.keys()) == {
        "dense",
        "bm25",
        "hybrid_rrf",
        "hybrid_reranked",
    }

    # Verify each configuration returned MultimodalSearchResult objects with provenance preserved
    for config_name, res_list in run.results.items():
        assert len(res_list) <= 2, f"Config {config_name} exceeded top_k=2"
        for item in res_list:
            assert isinstance(item, MultimodalSearchResult)
            assert item.element_id in {"txt-sec-1", "txt-sec-2", "txt-poison-1", "txt-poison-2"}
            assert item.source_filename == "sample_paper.pdf"
            assert isinstance(item.page_number, int)
            assert item.element_type == ElementType.TEXT
            assert isinstance(item.score, float)
            assert item.content != ""

    # Verify convenience property accessors
    assert run.dense_results == run.results["dense"]
    assert run.bm25_results == run.results["bm25"]
    assert run.hybrid_rrf_results == run.results["hybrid_rrf"]
    assert run.hybrid_reranked_results == run.results["hybrid_reranked"]

    # Verify latencies are recorded
    assert all(k in run.latencies for k in run.results.keys())
    assert all(run.latencies[k] >= 0.0 for k in run.latencies.keys())


def test_runner_configuration_subset_override(
    eval_pipeline: MultimodalRetrievalPipeline,
):
    """Verify runner can execute a specified subset of configurations."""
    dataset = get_retrieval_evaluation_dataset()
    q4 = dataset["Q4"]

    runner = RetrievalEvaluationRunner(pipeline=eval_pipeline)
    run = runner.run_question(
        question=q4,
        top_k=2,
        configurations=["dense", "bm25"],
    )

    assert set(run.results.keys()) == {"dense", "bm25"}
    assert len(run.results["dense"]) > 0
    assert len(run.results["bm25"]) > 0
    assert "hybrid_rrf" not in run.results
    assert "hybrid_reranked" not in run.results


def test_runner_run_dataset(
    eval_pipeline: MultimodalRetrievalPipeline,
):
    """Verify running multiple questions across a dataset."""
    dataset = get_retrieval_evaluation_dataset()
    multi_evidence = dataset.multi_evidence_questions  # Q1 and Q4

    runner = RetrievalEvaluationRunner(pipeline=eval_pipeline, default_top_k=2)
    runs = runner.run_dataset(multi_evidence)

    assert len(runs) == 2
    assert runs[0].question.id == "Q1"
    assert runs[1].question.id == "Q4"

    for run in runs:
        assert set(run.results.keys()) == {
            "dense",
            "bm25",
            "hybrid_rrf",
            "hybrid_reranked",
        }


def test_question_retrieval_run_serialization(
    eval_pipeline: MultimodalRetrievalPipeline,
):
    """Verify QuestionRetrievalRun.to_dict serialization."""
    dataset = get_retrieval_evaluation_dataset()
    q1 = dataset["Q1"]

    runner = RetrievalEvaluationRunner(pipeline=eval_pipeline, default_top_k=2)
    run = runner.run_question(q1)

    serialized = run.to_dict()
    assert serialized["question_id"] == "Q1"
    assert serialized["query"] == q1.query
    assert serialized["question_type"] == "conceptual"
    assert "latencies" in serialized
    assert "results" in serialized
    assert "dense" in serialized["results"]
    assert "hybrid_reranked" in serialized["results"]
    dense_first = serialized["results"]["dense"][0]
    assert "content" in dense_first
    assert "content_preview" in dense_first
    assert len(dense_first["content"]) > 0



def test_create_evaluation_pipeline_helper(
    eval_test_document: MultimodalDocument,
):
    """Verify factory create_evaluation_pipeline sets up FAISS, BM25, and Reranker properly."""
    fake_model = MockCrossEncoderModel()
    pipeline = create_evaluation_pipeline(
        document=eval_test_document,
        text_embedder=MockTextEmbedder(),
        visual_embedder=MockVisualEmbedder(),
        reranker=CrossEncoderReranker(model=fake_model),
    )

    assert isinstance(pipeline, MultimodalRetrievalPipeline)
    assert pipeline.has_bm25 is True
    assert pipeline.has_reranker is True
    assert isinstance(pipeline.index, MultimodalFAISSIndex)
    assert isinstance(pipeline.bm25_index, BM25Index)
    assert isinstance(pipeline.reranker, BaseReranker)


def test_retrieval_configuration_parsing():
    """Verify string parsing with aliases for RetrievalConfiguration."""
    assert RetrievalConfiguration.from_str("dense") == RetrievalConfiguration.DENSE
    assert RetrievalConfiguration.from_str("faiss") == RetrievalConfiguration.DENSE
    assert RetrievalConfiguration.from_str("bm25") == RetrievalConfiguration.BM25
    assert RetrievalConfiguration.from_str("sparse") == RetrievalConfiguration.BM25
    assert RetrievalConfiguration.from_str("hybrid_rrf") == RetrievalConfiguration.HYBRID_RRF
    assert RetrievalConfiguration.from_str("hybrid") == RetrievalConfiguration.HYBRID_RRF
    assert RetrievalConfiguration.from_str("hybrid_reranked") == RetrievalConfiguration.HYBRID_RERANKED
    assert RetrievalConfiguration.from_str("reranked") == RetrievalConfiguration.HYBRID_RERANKED

    with pytest.raises(ValueError, match="Unknown retrieval configuration"):
        RetrievalConfiguration.from_str("invalid_mode")


def test_build_sample_paper_document():
    """Verify build_sample_paper_document parses and assembles data/sample_paper.pdf."""
    from app.evaluation import build_sample_paper_document

    doc = build_sample_paper_document("data/sample_paper.pdf")
    assert isinstance(doc, MultimodalDocument)
    assert doc.source_filename == "sample_paper.pdf"
    assert len(doc.elements) > 0

    types = {e.element_type for e in doc.elements}
    assert ElementType.TEXT in types
    assert ElementType.IMAGE in types
    assert ElementType.CAPTION in types


def test_q6_caption_relationship_expands_to_actual_image():
    """Q6 text retrieval preserves the caption and adds its directly related image."""
    from app.evaluation import build_sample_paper_document

    doc = build_sample_paper_document("data/sample_paper.pdf")
    caption_id = "sample_paper.pdf#page_2_cap_1"
    image_id = "sample_paper.pdf#page_2_img_2"
    caption = doc.get_by_id(caption_id)
    image = doc.get_by_id(image_id)

    assert caption is not None
    assert image is not None
    assert caption.related_element_ids == [image_id]
    assert doc.get_related(caption_id) == [image]

    pipeline = create_evaluation_pipeline(doc, default_top_k=5)
    results = pipeline.retrieve_text(
        get_retrieval_evaluation_dataset()["Q6"].query,
        top_k=5,
    )
    result_ids = [result.element_id for result in results]
    assert caption_id in result_ids
    image_result = next(result for result in results if result.element_id == image_id)
    assert image_result.element_type == ElementType.IMAGE


def test_text_only_retrieval_without_relationships_is_unchanged(eval_pipeline):
    """Unrelated text-only results remain the same shape and modality."""
    results = eval_pipeline.retrieve_text("security", top_k=2)
    assert results
    assert all(result.element_type != ElementType.IMAGE for result in results)


def test_format_content_preview_and_results(eval_pipeline: MultimodalRetrievalPipeline):
    """Verify format_content_preview and format_question_results display strings."""
    from app.evaluation import format_content_preview, format_question_results

    preview = format_content_preview("Hello \n world \r multiple    spaces", max_len=20)
    assert preview == "Hello world multi..."
    assert len(preview) == 20

    binary_preview = format_content_preview(b"12345", max_len=20)
    assert "binary data" in binary_preview

    dataset = get_retrieval_evaluation_dataset()
    runner = RetrievalEvaluationRunner(pipeline=eval_pipeline, default_top_k=2)
    run = runner.run_question(dataset["Q1"])

    formatted = format_question_results(run, top_k=2)
    assert "[Q1]" in formatted
    assert "Dense FAISS" in formatted
    assert "BM25" in formatted
    assert "Hybrid RRF" in formatted
    assert "Hybrid RRF + Cross-Encoder" in formatted
