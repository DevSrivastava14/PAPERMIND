"""Execution entry point to run the retrieval evaluation baseline against data/sample_paper.pdf.

Runs all 8 evaluation questions through the four benchmark retrieval configurations:
1. Dense FAISS
2. BM25
3. Hybrid RRF
4. Hybrid RRF + Cross-Encoder

Outputs readable results with provenance, rank, score, content previews, and latency,
and saves the raw baseline results to JSON under data/evaluation/.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from app.chunking.text_chunker import chunk_document
from app.evaluation.models import EvaluationDataset, EvaluationQuestion
from app.evaluation.retrieval_dataset import get_retrieval_evaluation_dataset
from app.evaluation.runner import (
    ALL_RETRIEVAL_CONFIGURATIONS,
    QuestionRetrievalRun,
    RetrievalConfiguration,
    RetrievalEvaluationRunner,
    create_evaluation_pipeline,
)
from app.indexing.multimodal_models import MultimodalSearchResult
from app.ingestion.pdf_parser import parse_research_paper
from app.multimodal.caption_extractor import extract_captions
from app.multimodal.collection import MultimodalDocument, assemble_multimodal_document
from app.multimodal.equation_extractor import extract_equations
from app.multimodal.image_extractor import extract_images
from app.multimodal.table_extractor import extract_tables


DEFAULT_PDF_PATH: Path = Path("data/sample_paper.pdf")
DEFAULT_OUTPUT_JSON: Path = Path("data/evaluation/baseline_retrieval_results.json")
DEFAULT_TOP_K: int = 5


def build_sample_paper_document(pdf_path: Union[str, Path] = DEFAULT_PDF_PATH) -> MultimodalDocument:
    """Parses and assembles the multimodal document from the sample paper PDF."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    # 1. Text parsing & sliding-window chunking
    doc = parse_research_paper(path)
    chunks = chunk_document(doc)

    # 2. Multimodal extractors
    images = extract_images(path)
    caption_res = extract_captions(path, images=images)
    tables = extract_tables(path)
    equations = extract_equations(path)

    # 3. Assemble unified document
    return assemble_multimodal_document(
        text_chunks=chunks,
        images=images,
        captions=caption_res.captions,
        tables=tables,
        equations=equations,
        caption_matches=caption_res.matches,
        source_filename=path.name,
    )


def format_content_preview(content: Any, max_len: int = 80) -> str:
    """Truncates content to a single-line preview."""
    if isinstance(content, bytes):
        return f"<binary data: {len(content)} bytes>"
    text = str(content).replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    if len(text) > max_len:
        return text[:max_len - 3] + "..."
    return text


def format_question_results(run: QuestionRetrievalRun, top_k: int = DEFAULT_TOP_K) -> str:
    """Formats one question's four-configuration retrieval results for display."""
    lines: List[str] = []
    q = run.question
    lines.append("=" * 80)
    lines.append(f"[{q.id}] ({q.question_type.value.upper()})")
    lines.append(f"Query: {q.query}")
    lines.append(f"Reference Answer: {format_content_preview(q.reference_answer, max_len=120)}")
    lines.append("-" * 80)

    config_titles = [
        (RetrievalConfiguration.DENSE, "1. Dense FAISS"),
        (RetrievalConfiguration.BM25, "2. BM25"),
        (RetrievalConfiguration.HYBRID_RRF, "3. Hybrid RRF"),
        (RetrievalConfiguration.HYBRID_RERANKED, "4. Hybrid RRF + Cross-Encoder"),
    ]

    for config, title in config_titles:
        results = run.results.get(config.value, [])
        latency = run.latencies.get(config.value, 0.0)
        lines.append(f"\n  Configuration: {title}  |  Latency: {latency * 1000.0:.2f} ms")
        if not results:
            lines.append("    (No results returned)")
            continue

        lines.append(
            f"    {'Rank':<5} {'Type':<10} {'Page':<6} {'Element ID':<22} {'Score':<10} {'Content Preview'}"
        )
        lines.append(f"    {'-'*4:<5} {'-'*8:<10} {'-'*4:<6} {'-'*20:<22} {'-'*8:<10} {'-'*40}")

        for rank, item in enumerate(results[:top_k], start=1):
            etype = item.element_type.value if hasattr(item.element_type, "value") else str(item.element_type)
            score_str = f"{item.score:.4f}"
            preview = format_content_preview(item.content or (item.metadata.get("caption") if item.metadata else ""), max_len=60)
            lines.append(
                f"    {rank:<5} {etype:<10} p.{item.page_number:<4} {item.element_id:<22} {score_str:<10} {preview}"
            )

    lines.append("")
    return "\n".join(lines)


def run_retrieval_baseline(
    pdf_path: Union[str, Path] = DEFAULT_PDF_PATH,
    output_json_path: Union[str, Path] = DEFAULT_OUTPUT_JSON,
    top_k: int = DEFAULT_TOP_K,
    print_output: bool = True,
) -> Tuple[List[QuestionRetrievalRun], Dict[str, Any]]:
    """Runs the full evaluation baseline across all 8 questions and 4 configurations."""
    path = Path(pdf_path)
    out_path = Path(output_json_path)

    if print_output:
        print(f"Loading and assembling document from: {path.resolve()}")
    doc = build_sample_paper_document(path)
    if print_output:
        print(f"Assembled {len(doc.elements)} multimodal elements from {path.name}.")
        print("Initializing retrieval pipeline (FAISS + BM25 + CrossEncoder)...")

    pipeline = create_evaluation_pipeline(doc, default_top_k=top_k)
    dataset = get_retrieval_evaluation_dataset()
    runner = RetrievalEvaluationRunner(pipeline=pipeline, default_top_k=top_k)

    if print_output:
        print(f"Running baseline evaluation across {len(dataset)} questions...")
        print("Configurations: Dense FAISS | BM25 | Hybrid RRF | Hybrid RRF + Cross-Encoder\n")

    runs: List[QuestionRetrievalRun] = []
    for question in dataset:
        run = runner.run_question(question, top_k=top_k)
        runs.append(run)
        if print_output:
            print(format_question_results(run, top_k=top_k))

    # Calculate summary metrics (latencies)
    config_keys = [c.value for c in ALL_RETRIEVAL_CONFIGURATIONS]
    avg_latencies: Dict[str, float] = {}
    for c_key in config_keys:
        times = [r.latencies.get(c_key, 0.0) for r in runs if c_key in r.latencies]
        avg_latencies[c_key] = (sum(times) / len(times)) if times else 0.0

    # Build JSON payload
    payload: Dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pdf_path": str(path.as_posix()),
            "total_questions": len(runs),
            "top_k": top_k,
            "configurations": config_keys,
            "average_latency_ms": {k: round(v * 1000.0, 2) for k, v in avg_latencies.items()},
        },
        "questions": [
            {
                "id": r.question.id,
                "query": r.question.query,
                "question_type": r.question.question_type.value,
                "reference_answer": r.question.reference_answer,
                "special_notes": r.question.special_notes,
                "has_multiple_evidence": r.question.has_multiple_evidence,
                "gold_evidence": [e.to_dict() for e in r.question.gold_evidence],
                "latencies_ms": {k: round(v * 1000.0, 2) for k, v in r.latencies.items()},
                "results": {
                    config_name: [
                        {
                            "rank": idx + 1,
                            "element_id": item.element_id,
                            "element_type": item.element_type.value if hasattr(item.element_type, "value") else str(item.element_type),
                            "page_number": item.page_number,
                            "score": round(float(item.score), 6),
                            "source_filename": item.source_filename,
                            "content": str(item.content) if item.content is not None else "",
                            "content_preview": format_content_preview(item.content, max_len=120),
                        }

                        for idx, item in enumerate(res_list)
                    ]
                    for config_name, res_list in r.results.items()
                },
            }
            for r in runs
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if print_output:
        print("=" * 80)
        print("Baseline Evaluation Completed Successfully!")
        print(f"Total questions evaluated: {len(runs)}")
        print(f"Average latencies (ms): {payload['metadata']['average_latency_ms']}")
        print(f"Raw results saved to: {out_path.resolve()}")
        print("=" * 80)

    return runs, payload


def main() -> None:
    """CLI entry point for running the retrieval baseline."""
    run_retrieval_baseline()


if __name__ == "__main__":
    main()
