"""Tests for the Day 28 performance measurement utility."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from scripts.benchmark_pipeline import (
    DEFAULT_QUERY,
    MockBenchmarkGenerator,
    PipelineBenchmark,
    StageTiming,
    format_report_table,
)


def test_stage_timing_aggregation() -> None:
    timing = StageTiming(name="test_stage", description="A test stage")
    timing.samples_ms.extend([10.0, 20.0, 30.0])

    assert timing.count == 3
    assert timing.mean_ms == 20.0
    assert timing.min_ms == 10.0
    assert timing.max_ms == 30.0
    assert abs(timing.std_ms - 10.0) < 1e-6


def test_mock_benchmark_generator() -> None:
    gen = MockBenchmarkGenerator(answer="Test answer")
    assert gen.model_name == "mock-benchmark-generator"
    assert gen.generate("Test prompt") == "Test answer"


def test_pipeline_benchmark_runs_with_mock_generator() -> None:
    pdf_path = Path("data/sample_paper.pdf")
    if not pdf_path.is_file():
        pytest.skip("data/sample_paper.pdf not found in local workspace")

    benchmark = PipelineBenchmark(
        pdf_path=pdf_path,
        query="What is AI for Security?",
        top_k=2,
        generator=MockBenchmarkGenerator(),
        skip_llm=False,
    )

    report = benchmark.run_benchmark(runs=1)
    report_dict = report.to_dict()

    # Verify all 9 stages are present
    expected_stages = [
        "1_pdf_parsing",
        "2_text_embedding",
        "3_faiss_retrieval",
        "4_bm25_retrieval",
        "5_rrf_fusion",
        "6_relationship_expansion",
        "7_reranking",
        "8_llm_generation",
    ]
    for stage_key in expected_stages:
        assert stage_key in report_dict["stages"]
        assert report_dict["stages"][stage_key]["mean_ms"] >= 0.0
        assert len(report_dict["stages"][stage_key]["samples_ms"]) == 1

    # Stage 9: End to end query latency
    assert report_dict["end_to_end_query_ms"]["mean_ms"] > 0.0

    # Initialization timings
    assert "document_assembly_ms" in report_dict["initialization_ms"]
    assert "pipeline_index_build_ms" in report_dict["initialization_ms"]

    # Formatted output table
    table = format_report_table(report)
    assert "PAPERMIND PIPELINE PERFORMANCE BENCHMARK" in table
    assert "1_pdf_parsing" in table
    assert "9_end_to_end_query" in table


def test_pipeline_benchmark_skip_llm() -> None:
    pdf_path = Path("data/sample_paper.pdf")
    if not pdf_path.is_file():
        pytest.skip("data/sample_paper.pdf not found in local workspace")

    benchmark = PipelineBenchmark(
        pdf_path=pdf_path,
        query="What is AI for Security?",
        top_k=2,
        skip_llm=True,
    )

    report = benchmark.run_benchmark(runs=1)
    report_dict = report.to_dict()

    assert report_dict["stages"]["8_llm_generation"]["mean_ms"] == 0.0
    assert report_dict["stages"]["1_pdf_parsing"]["mean_ms"] > 0.0
    assert report_dict["stages"]["2_text_embedding"]["mean_ms"] > 0.0


def test_rerank_candidate_pool_default_is_10() -> None:
    from app.retrieval.multimodal_pipeline import DEFAULT_RERANK_CANDIDATE_POOL

    assert DEFAULT_RERANK_CANDIDATE_POOL == 10

