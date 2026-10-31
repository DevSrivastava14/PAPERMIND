# PaperMind — System Limitations and Boundary Analysis

> [!NOTE]
> This document details the architectural, algorithmic, operational, and evaluation limitations of PaperMind. Every item is verified against the actual Python source code (`app/`), Docker configuration, and saved benchmark artifacts (`data/evaluation/`).

---

## Table of Contents

1. [Architectural and Lifecycle Limitations](#1-architectural-and-lifecycle-limitations)
   - [Single-Paper-per-Query Ingestion Boundary](#single-paper-per-query-ingestion-boundary)
   - [In-Memory FAISS Index Rebuild Lifecycle](#in-memory-faiss-index-rebuild-lifecycle)
   - [Single-Hop Relationship Expansion](#single-hop-relationship-expansion)
2. [Multimodal and Model-Level Limitations](#2-multimodal-and-model-level-limitations)
   - [Text-Only Cross-Encoder Score Incompatibility with Images](#text-only-cross-encoder-score-incompatibility-with-images)
   - [CLIP Coarse Alignment and Resolution Constraints](#clip-coarse-alignment-and-resolution-constraints)
   - [Experimental Heuristic Score Blending](#experimental-heuristic-score-blending)
   - [No OCR or Visual Semantic Parsing of Rendered Content](#no-ocr-or-visual-semantic-parsing-of-rendered-content)
   - [Advanced Multimodal Document Encoders (ColPali) Deferred](#advanced-multimodal-document-encoders-colpali-deferred)
3. [Evaluation and Benchmark Limitations](#3-evaluation-and-benchmark-limitations)
   - [Single-Document Evaluation Benchmark (8 Questions)](#single-document-evaluation-benchmark-8-questions)
   - [Absence of Table and Equation Test Coverage](#absence-of-table-and-equation-test-coverage)
   - [DeepEval Incompleteness Due to Groq Rate Limits](#deepeval-incompleteness-due-to-groq-rate-limits)
   - [Evaluation Metrics Scope](#evaluation-metrics-scope)
4. [Operational and Production Limitations](#4-operational-and-production-limitations)
   - [CPU Reranking Latency Ceiling](#cpu-reranking-latency-ceiling)
   - [Absence of Distributed Tracing (LangSmith Deferred)](#absence-of-distributed-tracing-langsmith-deferred)
   - [No Persistent Vector Database Storage](#no-persistent-vector-database-storage)
5. [Summary Classification Matrix](#5-summary-classification-matrix)

---

## 1. Architectural and Lifecycle Limitations

### Single-Paper-per-Query Ingestion Boundary

- **What it is:** The FastAPI endpoint (`POST /query`) accepts a single `pdf_path` string per request. All document parsing, text chunking, visual extraction, indexing, and retrieval are strictly scoped to that individual PDF.
- **Why it exists:** PaperMind was specifically engineered for deep, grounded exploration of a single complex technical paper with high-fidelity citation traceability, rather than a broad multi-document corpus search engine.
- **Practical impact:** Users cannot issue cross-document comparative queries (e.g., comparing methodology across 10 papers) in a single request. Multi-paper workflows require separate API calls per paper.
- **Classification:** **Intentional scope decision**.

### In-Memory FAISS Index Rebuild Lifecycle

- **What it is:** Dense vector indexes (`FAISSIndex`, `MultimodalFAISSIndex`) and the sparse BM25 inverted index (`BM25Index`) are instantiated entirely in memory during request execution and discarded upon request completion.
- **Why it exists:** In-memory index generation for an 8-page paper is exceptionally fast (PDF parsing ~51.3 ms, embedding ~17.3 ms, FAISS retrieval ~0.35 ms). Rebuilding indexes eliminates index-staleness bugs, state leakage across concurrent queries, and disk serialization overhead.
- **Practical impact:** Each query re-parses and re-embeds the document. While negligible for an 8-page paper (<100 ms ingestion overhead), querying a 100-page book or re-querying the same paper hundreds of times incurs redundant compute.
- **Classification:** **Intentional scope decision** (stateless local design; vector persistence deferred).

### Single-Hop Relationship Expansion

- **What it is:** Relationship expansion traverses exactly one edge between explicitly related multimodal elements (`caption -> image`).
- **Why it exists:** Designed to solve the specific multimodal failure mode where a figure caption is retrieved by text search but the underlying diagram (`IMAGE`) is excluded.
- **Practical impact:** Multi-hop references (e.g., in-text paragraph referencing Section II, which references Figure 2, which references an appendix table) cannot be traversed automatically. Only elements with direct, pre-computed `related_element_ids` are expanded.
- **Classification:** **Intentional scope decision** (preventing graph explosion and topic drift).

---

## 2. Multimodal and Model-Level Limitations

### Text-Only Cross-Encoder Score Incompatibility with Images

- **What it is:** The standard cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is a text-pair model. When evaluating an image element, the candidate text extractor falls back to empty strings or metadata placeholders, producing severely penalized scores (e.g., `-8.65` raw logit) that eject images from the final top-k.
- **Why it exists:** Sentence Transformers cross-encoders are sequence-classification language models, not vision-language models; they cannot process raw pixel matrices.
- **Practical impact:** In text-only reranked retrieval, images are dropped during reranking even if retrieved during dense or sparse stages.
- **Classification:** **Model limitation mitigated by orchestration**; `app/orchestration/langgraph_graph.py` automatically routes documents containing images to `MultimodalReranker`, preserving images through visual CLIP scoring.

### CLIP Coarse Alignment and Resolution Constraints

- **What it is:** Visual retrieval and multimodal reranking use `sentence-transformers/clip-ViT-B-32`. CLIP computes cosine similarity between a 512-dimensional text-query embedding and a 512-dimensional image embedding.
- **Why it exists:** ViT-B-32 is lightweight and runs comfortably on CPU, but was trained on natural web images and short alt-text pairs.
- **Practical impact:** Fine-grained technical diagrams, dense flowcharts, and small textual labels inside figures are poorly represented in global CLIP embedding space. CLIP can identify that a diagram relates to "AI architecture", but struggles to discriminate between subtle algorithmic sub-diagrams without caption context.
- **Classification:** **Model capability ceiling** (open research problem in vision-language models).

### Experimental Heuristic Score Blending

- **What it is:** In `MultimodalReranker`, final scores are blended via fixed linear weights:
  - Text-like elements: `0.80 * CE_norm + 0.20 * RRF_norm`
  - Image elements: `0.80 * CLIP_norm + 0.20 * RRF_norm`
- **Why it exists:** Cross-encoder logits, CLIP cosine similarities, and RRF scores exist on completely different scales and distributions.
- **Practical impact:** While min-max normalization with near-zero guards and single-element default values (`0.5`) stabilizes ranking, the `0.80 / 0.20` ratio is an empirical heuristic tuned on benchmark questions rather than a mathematically calibrated probability.
- **Classification:** **Unresolved work / active engineering trade-off** (statistical Platt scaling or learned ranking models deferred).

### No OCR or Visual Semantic Parsing of Rendered Content

- **What it is:** Raster images embedded within the PDF are extracted as binary JPEG/PNG assets via PyMuPDF. No OCR (e.g., Tesseract) or visual document parser is applied to the image pixels.
- **Why it exists:** Avoiding heavy native C++ OCR dependencies and keeping ingestion fast and lightweight.
- **Practical impact:** Text embedded inside diagram boxes, axis labels on plots, and text-rendered tables cannot be indexed by BM25 or dense text FAISS unless captured by adjacent caption text.
- **Classification:** **Intentional scope boundary**.

### Advanced Multimodal Document Encoders (ColPali) Deferred

- **What it is:** Modern vision-language document retrieval models like ColPali (Paligemma-based multi-vector late interaction) are not integrated.
- **Why it exists:** ColPali models require massive multi-gigabyte download footprints, high GPU memory (VRAM), and substantial cold-start initialization latency that would prevent lightweight local Docker/CPU reproduction.
- **Practical impact:** Retrieval relies on hybrid text + CLIP late fusion rather than native end-to-end page image embeddings.
- **Classification:** **Intentional scope decision** (cold-start latency and hardware requirements unquantified on commodity CPU).

---

## 3. Evaluation and Benchmark Limitations

### Single-Document Evaluation Benchmark (8 Questions)

- **What it is:** The deterministic retrieval benchmark consists of 8 curated questions evaluated against `data/sample_paper.pdf`.
- **Why it exists:** Allows rigorous, hand-verified gold chunk definitions and deep diagnostic tracing across multiple pipeline iterations (Days 15–30).
- **Practical impact:** While retrieval mechanics (BM25, RRF, Cross-Encoder, LangGraph orchestration) are robustly tested across 100+ unit and integration tests, statistical generalization across a diverse corpus of hundreds of disparate document layouts has not been benchmarked.
- **Classification:** **Dataset scope limitation**.

### Absence of Table and Equation Test Coverage

- **What it is:** The canonical `data/sample_paper.pdf` contains 0 extracted table elements and 0 extracted standalone equation elements (`day23_lightweight_baseline.json` records `"valid_measurement": false, "status": "not_runnable"`).
- **Why it exists:** The sample paper relies on prose descriptions and visual diagrams rather than markdown tables or LaTeX formula blocks.
- **Practical impact:** Although extraction, tokenization (`d_k`, `bleu-4`), and BM25 indexing for `TABLE` and `EQUATION` elements are thoroughly verified in unit tests (`tests/test_bm25_index.py`), real-world end-to-end retrieval performance on equation-heavy papers (e.g., *Attention Is All You Need*) has not been quantified.
- **Classification:** **Benchmark coverage gap**.

### DeepEval Incompleteness Due to Groq Rate Limits

- **What it is:** In `data/evaluation/day25_deepeval_results.json`, only Question 1 produced completed DeepEval scores (faithfulness 1.0, answer relevancy 1.0, contextual precision 1.0, contextual recall 1.0, contextual relevancy 0.3529). Q6 and the INSUFFICIENT test cases failed due to Groq free-tier rate limits (~8,000 TPM limit exhausted during multi-turn LLM judge evaluations) and judge JSON formatting errors.
- **Why it exists:** LLM-as-a-judge frameworks serialize full context nodes into multiple prompt passes, consuming substantial token quotas.
- **Practical impact:** End-to-end answer quality has quantitative confirmation on Q1, but multimodal answer evaluation remains incomplete.
- **Classification:** **Infrastructure/quota constraint**.

### Evaluation Metrics Scope

- **What it is:** Evaluation is focused on MRR, partial recall, complete recall, and DeepEval LLM-as-judge. Classical generation metrics like ROUGE, BLEU, BERTScore, or Ragas are not integrated.
- **Why it exists:** For technical Q&A, n-gram lexical overlap (BLEU/ROUGE) correlates poorly with factual truthfulness, while MRR directly evaluates retrieval fidelity.
- **Practical impact:** Academic comparative tables against non-RAG extractive baselines are not provided.
- **Classification:** **Intentional design choice**.

---

## 4. Operational and Production Limitations

### CPU Reranking Latency Ceiling

- **What it is:** On a standard CPU, running `ms-marco-MiniLM-L-6-v2` cross-attention over candidates is the dominant latency bottleneck.
- **Why it exists:** Cross-encoders pass concatenated `(query, passage)` pairs through all 6 transformer layers.
- **Practical impact:** In Day 28, reducing candidate pool from 20 to 10 brought reranking latency down from 786.27 ms to 452.24 ms (-42.5%), enabling sub-second local execution (979 ms local path). However, scaling candidate pools to 50 or 100 on CPU would push response times to multiple seconds. Production deployment requires GPU acceleration or model quantization (ONNX/TensorRT).
- **Classification:** **Hardware / compute constraint**.

### Absence of Distributed Tracing (LangSmith Deferred)

- **What it is:** PaperMind uses Python standard-library structured JSON logging (`papermind.api`, `papermind.retrieval`, `papermind.generation`) rather than a hosted observability platform like LangSmith or OpenTelemetry.
- **Why it exists:** LangSmith requires external service credentials, network egress, and third-party dependencies without improving retrieval quality for a single-service architecture.
- **Practical impact:** Traces are inspected via log streams (`docker compose logs`) or stdout, rather than an interactive graphical UI.
- **Classification:** **Intentional scope decision** (deferred pending multi-service requirement).

### No Persistent Vector Database Storage

- **What it is:** No standalone vector database server (Qdrant, Milvus, Chroma, pgvector) is run.
- **Why it exists:** Docker Compose keeps resource consumption minimal (single container, 8000 port). Hugging Face model weights are persisted in the `hf_cache` Docker volume, but vector indexes remain ephemeral.
- **Practical impact:** Storage is simple and self-contained, but pre-indexing an institutional library for instant querying is not supported without re-indexing.
- **Classification:** **Intentional architectural choice**.

---

## 5. Summary Classification Matrix

| Limitation | Category | Impact | Status |
|---|---|---|---|
| Single-paper-per-query | Architecture | No cross-document queries | Intentional scope |
| In-memory FAISS lifecycle | Storage | Redundant index building for repeated queries | Intentional scope |
| 1-hop relationship expansion | Graph Retrieval | Multi-hop reasoning not traversed | Intentional scope |
| Text cross-encoder image blindspot | Modality | Text-only model penalizes raw images | Mitigated via `MultimodalReranker` |
| CLIP coarse semantic alignment | Model | Dense technical diagram text missed | Model capability ceiling |
| Fixed 80/20 heuristic scoring | Algorithmic | Empirical rather than calibrated probabilities | Unresolved / future tuning |
| No image OCR | Ingestion | Text inside diagrams unindexed | Intentional scope |
| ColPali VLM deferred | Model / Infra | High cold-start latency & GPU needs avoided | Intentional scope |
| 8-question benchmark size | Evaluation | Single-paper benchmark; corpus scale unmeasured | Dataset scope constraint |
| Zero table/equation test samples | Evaluation | Extraction works in unit tests; end-to-end unmeasured | Benchmark data gap |
| DeepEval Groq rate limits | Evaluation | Only Q1 complete; Q6/INSUFFICIENT rate-limited | Infrastructure/quota constraint |
| CPU reranking latency | Performance | Reranking candidate pool capped at 10 for latency | Hardware constraint (GPU needed for pool>30) |
| LangSmith deferred | Observability | Structured JSON logs used instead of hosted UI | Intentional scope |
| No persistent vector DB | Infrastructure | Zero vector DB overhead; ephemeral indexes | Intentional scope |

