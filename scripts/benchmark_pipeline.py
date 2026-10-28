"""Lightweight performance measurement utility for the PaperMind RAG pipeline.

Measures the 9 separable stages of the PaperMind pipeline using monotonic
high-resolution timers (time.perf_counter()):
1. PDF parsing
2. text embedding
3. FAISS retrieval
4. BM25 retrieval
5. RRF fusion
6. relationship expansion
7. text/multimodal reranking
8. LLM generation
9. end-to-end query latency

Ensures model and API initializations are tracked separately and not counted
as repeated per-query latency.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Union

# Ensure project root is on sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import find_dotenv, load_dotenv

from app.evaluation.evaluate_retrieval import build_sample_paper_document
from app.evaluation.runner import create_evaluation_pipeline
from app.generation.answer_generator import AnswerGenerator, build_answer_generator
from app.generation.base import BaseGenerator
from app.ingestion.pdf_parser import parse_research_paper
from app.retrieval.multimodal_pipeline import (
    DEFAULT_RERANK_CANDIDATE_POOL,
    MultimodalRetrievalPipeline,
)
from app.retrieval.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion

DEFAULT_PDF_PATH: str = "data/sample_paper.pdf"
DEFAULT_QUERY: str = 'What is the difference between "AI for Security" and "Security for AI"?'
DEFAULT_TOP_K: int = 5
DEFAULT_RUNS: int = 3


class MockBenchmarkGenerator(BaseGenerator):
    """Deterministic mock generator for offline benchmarking and tests."""

    def __init__(self, answer: str = "Mock answer for benchmark timing.") -> None:
        self._answer = answer

    @property
    def model_name(self) -> str:
        return "mock-benchmark-generator"

    def generate(self, prompt: str) -> str:
        return self._answer


@dataclass
class StageTiming:
    """Stores repeated measurements for a single stage in milliseconds."""

    name: str
    description: str
    samples_ms: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples_ms)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.samples_ms) if self.samples_ms else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms) if self.samples_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.samples_ms) if self.samples_ms else 0.0

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.samples_ms) if len(self.samples_ms) > 1 else 0.0


@dataclass
class BenchmarkReport:
    """Full performance benchmark report containing initialization and per-stage timings."""

    pdf_path: str
    query: str
    top_k: int
    num_runs: int
    initialization_ms: Dict[str, float]
    stages: Dict[str, StageTiming]
    e2e_query_ms: StageTiming
    pipeline_total_measured_ms: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pdf_path": self.pdf_path,
            "query": self.query,
            "top_k": self.top_k,
            "num_runs": self.num_runs,
            "initialization_ms": self.initialization_ms,
            "stages": {
                k: {
                    "name": v.name,
                    "description": v.description,
                    "mean_ms": round(v.mean_ms, 3),
                    "min_ms": round(v.min_ms, 3),
                    "max_ms": round(v.max_ms, 3),
                    "std_ms": round(v.std_ms, 3),
                    "samples_ms": [round(s, 3) for s in v.samples_ms],
                }
                for k, v in self.stages.items()
            },
            "end_to_end_query_ms": {
                "name": self.e2e_query_ms.name,
                "description": self.e2e_query_ms.description,
                "mean_ms": round(self.e2e_query_ms.mean_ms, 3),
                "min_ms": round(self.e2e_query_ms.min_ms, 3),
                "max_ms": round(self.e2e_query_ms.max_ms, 3),
                "std_ms": round(self.e2e_query_ms.std_ms, 3),
                "samples_ms": [round(s, 3) for s in self.e2e_query_ms.samples_ms],
            },
            "pipeline_total_measured_ms": round(self.pipeline_total_measured_ms, 3),
            "notes": self.notes,
        }


class PipelineBenchmark:
    """Measures latency of each separable stage in the PaperMind RAG pipeline."""

    STAGE_DESCRIPTIONS = {
        "1_pdf_parsing": "Extract pages and plain text metadata from PDF file (PyMuPDF)",
        "2_text_embedding": "Encode query text into 384-d dense vector (SentenceTransformer)",
        "3_faiss_retrieval": "Dense cosine similarity vector search over 384-d FAISS index",
        "4_bm25_retrieval": "Lexical inverted index tokenization and BM25 scoring",
        "5_rrf_fusion": "Reciprocal Rank Fusion merging dense and sparse ranked candidates",
        "6_relationship_expansion": "Traverse inter-element links (figures <-> captions <-> text)",
        "7_reranking": "Cross-encoder neural relevance scoring and re-sorting",
        "8_llm_generation": "Context prompt assembly and LLM response generation",
        "9_end_to_end_query": "Complete query -> retrieval -> rerank -> generation request path",
    }

    def __init__(
        self,
        pdf_path: Union[str, Path] = DEFAULT_PDF_PATH,
        query: str = DEFAULT_QUERY,
        top_k: int = DEFAULT_TOP_K,
        generator: Optional[BaseGenerator] = None,
        skip_llm: bool = False,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.query = query
        self.top_k = top_k
        self.generator = generator
        self.skip_llm = skip_llm

        self.initialization_ms: Dict[str, float] = {}
        self.document: Optional[MultimodalDocument] = None
        self.pipeline: Optional[MultimodalRetrievalPipeline] = None
        self.answer_generator: Optional[AnswerGenerator] = None

    def initialize_and_warmup(self) -> None:
        """Initializes models and builds indices once so cold-start overhead is not counted in per-query stages."""
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path)

        # 1. Measure one-time document parsing and multimodal assembly
        t0 = time.perf_counter()
        self.document = build_sample_paper_document(self.pdf_path)
        self.initialization_ms["document_assembly_ms"] = (time.perf_counter() - t0) * 1000.0

        # 2. Build pipeline (loads embedding model and cross-encoder reranker, indexes vectors and BM25)
        t0 = time.perf_counter()
        self.pipeline = create_evaluation_pipeline(
            document=self.document,
            default_top_k=self.top_k,
        )
        self.initialization_ms["pipeline_index_build_ms"] = (time.perf_counter() - t0) * 1000.0

        # 3. Resolve generator
        if not self.skip_llm:
            if self.generator is None:
                groq_key = os.getenv("GROQ_API_KEY")
                if groq_key:
                    from app.generation.groq_generator import GroqGenerator

                    t0 = time.perf_counter()
                    self.generator = GroqGenerator()
                    self.initialization_ms["generator_init_ms"] = (time.perf_counter() - t0) * 1000.0
                else:
                    self.generator = MockBenchmarkGenerator(
                        answer="Mock grounded answer (no GROQ_API_KEY configured)."
                    )
                    self.initialization_ms["generator_init_ms"] = 0.0

            self.answer_generator = build_answer_generator(generator=self.generator)

        # 4. Warm-up run for neural models (ensures PyTorch weights and caches are hot)
        t0 = time.perf_counter()
        _ = self.pipeline.text_embedder.embed_texts(["Warmup sentence for embedder"])
        if self.pipeline.reranker is not None:
            # Warm up cross-encoder with dummy candidate search result if available
            dummy_candidates = self.pipeline.index.search_text("warmup", top_k=1)
            if dummy_candidates:
                self.pipeline.reranker.rerank("warmup", dummy_candidates, top_k=1)
        self.initialization_ms["model_warmup_ms"] = (time.perf_counter() - t0) * 1000.0

    def measure_pdf_parsing(self) -> float:
        """Stage 1: Measures raw PDF parsing latency via parse_research_paper."""
        t0 = time.perf_counter()
        _ = parse_research_paper(self.pdf_path)
        return (time.perf_counter() - t0) * 1000.0

    def run_separable_query_stages(self) -> Dict[str, float]:
        """Measures each per-query stage where they are individually separable."""
        if self.pipeline is None or self.document is None:
            raise RuntimeError("Pipeline must be initialized before running query stages.")

        timings: Dict[str, float] = {}

        # Stage 2: text embedding
        t0 = time.perf_counter()
        query_vectors = self.pipeline.text_embedder.embed_texts([self.query])
        query_vector = query_vectors[0]
        timings["2_text_embedding"] = (time.perf_counter() - t0) * 1000.0

        candidate_pool_k = max(self.top_k, DEFAULT_RERANK_CANDIDATE_POOL)

        # Stage 3: FAISS retrieval (pure vector search on precomputed embedding, no expansion)
        t0 = time.perf_counter()
        dense_raw = self.pipeline.index.search_text_by_vector(
            query_vector=query_vector,
            top_k=candidate_pool_k,
        )
        timings["3_faiss_retrieval"] = (time.perf_counter() - t0) * 1000.0

        # Stage 4: BM25 retrieval (pure sparse lexical search, no expansion)
        t0 = time.perf_counter()
        bm25_raw = self.pipeline.bm25_index.search(
            query=self.query,
            top_k=candidate_pool_k,
        )
        timings["4_bm25_retrieval"] = (time.perf_counter() - t0) * 1000.0

        # Stage 5: RRF fusion (pure reciprocal rank fusion calculation)
        t0 = time.perf_counter()
        rrf_fused = reciprocal_rank_fusion(
            ranked_lists=[dense_raw, bm25_raw],
            k=DEFAULT_RRF_K,
            top_k=candidate_pool_k,
        )
        timings["5_rrf_fusion"] = (time.perf_counter() - t0) * 1000.0

        # Stage 6: relationship expansion (pure traversal of related_element_ids)
        t0 = time.perf_counter()
        expanded_candidates = self.pipeline._expand_related_results(rrf_fused)
        timings["6_relationship_expansion"] = (time.perf_counter() - t0) * 1000.0

        # Stage 7: text/multimodal reranking (pure cross-encoder inference and re-ranking)
        t0 = time.perf_counter()
        reranked = self.pipeline.reranker.rerank(
            query=self.query,
            candidates=expanded_candidates,
            top_k=self.top_k,
        )
        timings["7_reranking"] = (time.perf_counter() - t0) * 1000.0

        # Context assembly (format retrieved elements into prompt-ready context)
        t0 = time.perf_counter()
        context = self.pipeline.assembler.assemble(
            search_results=reranked,
            document=self.document,
        )
        timings["context_assembly"] = (time.perf_counter() - t0) * 1000.0

        # Stage 8: LLM generation
        if not self.skip_llm and self.answer_generator is not None:
            t0 = time.perf_counter()
            _ = self.answer_generator.generate_from_context(
                query=self.query,
                context=context,
            )
            timings["8_llm_generation"] = (time.perf_counter() - t0) * 1000.0
        else:
            timings["8_llm_generation"] = 0.0

        return timings

    def measure_end_to_end_query(self) -> float:
        """Stage 9: Measures end-to-end query latency through the uninstrumented pipeline."""
        if self.pipeline is None or self.document is None:
            raise RuntimeError("Pipeline must be initialized before running query.")

        t0 = time.perf_counter()
        context = self.pipeline.hybrid_retrieve_reranked_and_assemble(
            query=self.query,
            top_k=self.top_k,
            document=self.document,
        )
        if not self.skip_llm and self.answer_generator is not None:
            _ = self.answer_generator.generate_from_context(
                query=self.query,
                context=context,
            )
        return (time.perf_counter() - t0) * 1000.0

    def run_benchmark(self, runs: int = DEFAULT_RUNS) -> BenchmarkReport:
        """Executes initialization, warmup, and the specified number of benchmark iterations."""
        if runs < 1:
            raise ValueError(f"runs must be at least 1, got {runs}")

        # One-time initialization and warmup
        self.initialize_and_warmup()

        # Initialize storage
        stages_data: Dict[str, StageTiming] = {
            key: StageTiming(name=key, description=self.STAGE_DESCRIPTIONS[key])
            for key in [
                "1_pdf_parsing",
                "2_text_embedding",
                "3_faiss_retrieval",
                "4_bm25_retrieval",
                "5_rrf_fusion",
                "6_relationship_expansion",
                "7_reranking",
                "8_llm_generation",
            ]
        }
        # Include context assembly as an informative stage
        stages_data["context_assembly"] = StageTiming(
            name="context_assembly",
            description="Format retrieved elements with citations into prompt-ready context",
        )

        e2e_data = StageTiming(
            name="9_end_to_end_query",
            description=self.STAGE_DESCRIPTIONS["9_end_to_end_query"],
        )

        # Run benchmark iterations
        for _ in range(runs):
            # Stage 1: PDF parsing
            parse_ms = self.measure_pdf_parsing()
            stages_data["1_pdf_parsing"].samples_ms.append(parse_ms)

            # Stages 2 - 8
            query_timings = self.run_separable_query_stages()
            for k, val in query_timings.items():
                if k in stages_data:
                    stages_data[k].samples_ms.append(val)

            # Stage 9: End-to-end uninstrumented query
            e2e_ms = self.measure_end_to_end_query()
            e2e_data.samples_ms.append(e2e_ms)

        # Sum of per-query stages (embedding + FAISS + BM25 + RRF + expansion + reranking + assembly + LLM)
        per_query_stage_keys = [
            "2_text_embedding",
            "3_faiss_retrieval",
            "4_bm25_retrieval",
            "5_rrf_fusion",
            "6_relationship_expansion",
            "7_reranking",
            "context_assembly",
            "8_llm_generation",
        ]
        pipeline_total_ms = sum(stages_data[k].mean_ms for k in per_query_stage_keys)

        notes = [
            "Model weights, index building, and API client setup were warmed up once during initialization.",
            "PDF parsing (Stage 1) is a document ingestion cost; it does not recur on per-query serving if pre-indexed.",
            "In existing production code, pipeline.retrieve_text and pipeline.retrieve_sparse internally bundle relationship expansion, so individual search vs expansion is only separable by invoking the underlying sub-components.",
        ]

        return BenchmarkReport(
            pdf_path=str(self.pdf_path),
            query=self.query,
            top_k=self.top_k,
            num_runs=runs,
            initialization_ms=self.initialization_ms,
            stages=stages_data,
            e2e_query_ms=e2e_data,
            pipeline_total_measured_ms=pipeline_total_ms,
            notes=notes,
        )


def format_report_table(report: BenchmarkReport) -> str:
    """Formats the benchmark report into a clean, human-readable ASCII table."""
    lines: List[str] = []
    lines.append("=" * 90)
    lines.append("               PAPERMIND PIPELINE PERFORMANCE BENCHMARK (DAY 28)")
    lines.append("=" * 90)
    lines.append(f"PDF Document   : {report.pdf_path}")
    lines.append(f"Query          : {report.query}")
    lines.append(f"Top-K Results  : {report.top_k}")
    lines.append(f"Benchmark Runs : {report.num_runs} iterations (reported as mean +/- std)")
    lines.append("-" * 90)

    lines.append("ONE-TIME INITIALIZATION & WARMUP (Excluded from per-query measurements):")
    for k, v in report.initialization_ms.items():
        lines.append(f"  - {k:<28}: {v:9.2f} ms")
    lines.append("-" * 90)

    lines.append("STAGE-BY-STAGE MEASUREMENTS (Monotonic time.perf_counter()):")
    header = f"{'#':<3} {'Stage Name':<26} {'Mean (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'Std (ms)':>10}"
    lines.append(header)
    lines.append("-" * 90)

    stage_order = [
        ("1", "1_pdf_parsing"),
        ("2", "2_text_embedding"),
        ("3", "3_faiss_retrieval"),
        ("4", "4_bm25_retrieval"),
        ("5", "5_rrf_fusion"),
        ("6", "6_relationship_expansion"),
        ("7", "7_reranking"),
        ("-", "context_assembly"),
        ("8", "8_llm_generation"),
    ]

    for num, key in stage_order:
        timing = report.stages.get(key)
        if timing is not None:
            lines.append(
                f"{num:<3} {timing.name:<26} {timing.mean_ms:>10.2f} {timing.min_ms:>10.2f} {timing.max_ms:>10.2f} {timing.std_ms:>10.2f}"
            )

    lines.append("-" * 90)
    e2e = report.e2e_query_ms
    lines.append(
        f"{'9':<3} {e2e.name:<26} {e2e.mean_ms:>10.2f} {e2e.min_ms:>10.2f} {e2e.max_ms:>10.2f} {e2e.std_ms:>10.2f}"
    )
    lines.append(f"    (Sum of isolated per-query stages 2-8: {report.pipeline_total_measured_ms:.2f} ms)")
    lines.append("=" * 90)

    lines.append("STAGE DESCRIPTIONS & REPRESENTATION:")
    for num, key in stage_order:
        timing = report.stages.get(key)
        if timing:
            lines.append(f"  [{num}] {timing.name:<24}: {timing.description}")
    lines.append(f"  [9] {e2e.name:<24}: {e2e.description}")
    lines.append("-" * 90)

    lines.append("PIPELINE ISOLATION OBSERVATIONS:")
    for note in report.notes:
        lines.append(f"  * {note}")
    lines.append("=" * 90)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run minimal performance measurement instrumentation for PaperMind."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=DEFAULT_PDF_PATH,
        help="Path to PDF paper (default: data/sample_paper.pdf)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Representative query to benchmark",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-K retrieval limit (default: 5)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Number of measurement iterations (default: 3)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM generation to measure retrieval-only pipeline",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use deterministic mock LLM for offline testing without Groq calls",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to output raw measurements in JSON format",
    )

    args = parser.parse_args()

    generator: Optional[BaseGenerator] = None
    if args.mock_llm:
        generator = MockBenchmarkGenerator()

    benchmark = PipelineBenchmark(
        pdf_path=args.pdf,
        query=args.query,
        top_k=args.top_k,
        generator=generator,
        skip_llm=args.skip_llm,
    )

    print(f"Running PaperMind benchmark with {args.runs} iteration(s)...")
    report = benchmark.run_benchmark(runs=args.runs)
    formatted = format_report_table(report)
    print(formatted)

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nSaved JSON results to: {out_path}")


if __name__ == "__main__":
    main()
