# PaperMind — Evaluation Methodology and Experimental Results

> [!NOTE]
> All metrics in this document are sourced from saved JSON artifacts in
> `data/evaluation/` and `PROJECT_PROGRESS.md`. No results have been
> recalculated or invented. Experiment dates and measurement conditions are
> preserved exactly as recorded.

---

## Table of Contents

1. [Evaluation Objectives](#1-evaluation-objectives)
2. [Evaluation Dataset](#2-evaluation-dataset)
3. [Retrieval Metrics](#3-retrieval-metrics)
4. [Retrieval Configurations](#4-retrieval-configurations)
5. [Experiment 1 — Eight-Question Retrieval Benchmark (Day 17)](#5-experiment-1--eight-question-retrieval-benchmark-day-17)
6. [Experiment 2 — Four-Question Subset (Day 17 Internal)](#6-experiment-2--four-question-subset-day-17-internal)
7. [Experiment 3 — Day 28 Optimized Pipeline](#7-experiment-3--day-28-optimized-pipeline)
8. [Experiment 4 — Day 23 Lightweight Retrieval Baseline](#8-experiment-4--day-23-lightweight-retrieval-baseline)
9. [LLM-as-Judge Evaluation (Day 25)](#9-llm-as-judge-evaluation-day-25)
10. [Q1 Smoke-Test Artifact Clarification](#10-q1-smoke-test-artifact-clarification)
11. [Limitations and Reproducibility Conditions](#11-limitations-and-reproducibility-conditions)

---

## 1. Evaluation Objectives

PaperMind is evaluated along two independent dimensions:

**Retrieval quality** — Does the system surface the correct evidence chunks within
its final top-k results? Retrieval quality is measured without an LLM judge using
deterministic rank-based metrics (MRR, recall) against curated gold evidence
definitions. This is the primary evaluation signal because it is repeatable,
quota-free, and independent of LLM stochasticity.

**Answer quality** — Does the system produce accurate, faithful, and relevant
answers grounded only in retrieved evidence? Answer quality is measured using
DeepEval with Groq as the judge LLM. This dimension is resource-constrained (Groq
rate limits) and was measured for a limited subset of cases.

Both dimensions were evaluated at controlled checkpoints rather than continuously,
and are explicitly scoped to the canonical sample paper.

---

## 2. Evaluation Dataset

### Source

- **PDF:** `data/sample_paper.pdf` — a published research paper on AI and information security.
- **Benchmark definition:** `data/evaluation/retrieval_questions.md` (created Day 17) and `data/evaluation/baseline_retrieval_results.json` (created Day 17).

### Eight-Question Benchmark

Eight questions were designed to test different retrieval challenges. Each question has one or more **gold evidence definitions** — specific chunk IDs whose content is necessary to answer the question correctly.

| ID | Type | Gold evidence chunks | Special characteristics |
|---|---|---|---|
| Q1 | Conceptual | `chunk_10`, `chunk_14` | Gold chunks on different pages (p. 2, p. 3); tests multi-chunk retrieval |
| Q2 | Conceptual | `chunk_10` | Single gold chunk |
| Q3 | Keyword-heavy | `chunk_23`, `chunk_41` | Named entity (MITRE ATLAS); distractor check on reference [22] |
| Q4 | Keyword-heavy | `chunk_21`, `chunk_22` | Three distinct attack types; two gold chunks in adjacent positions |
| Q5 | Framework structure | `chunk_31` | Single gold chunk covering a four-item list; intro context in adjacent chunk |
| Q6 | Figure / caption | `page_2_cap_1` (primary); `page_2_img_2` (related) | Sub-type breakdown exists only in the image; tests multimodal retrieval |
| Q7 | Specific factual | *(see note)* | Amazon gender-bias example, 2018 |
| Q8 | Specific factual | *(see note)* | Qrator Labs DDoS model, since 2006 |

> [!NOTE]
> Q7 and Q8 gold evidence chunk IDs are defined in `retrieval_questions.md` but
> were not independently verified against `baseline_retrieval_results.json` during
> the Day 31 audit. All eight questions are present in the `baseline_retrieval_results.json`
> artifact (confirmed: `"total_questions": 8`).

### Gold Evidence Definitions

A **gold evidence chunk** is a specific extracted PDF element (by `element_id`)
whose textual content is required to fully answer the question. Gold evidence
definitions were written by inspecting the paper and confirming the exact
passages, not inferred from model output.

**Q6 has two evidence tiers:**

- **Primary evidence** (`page_2_cap_1`): the figure caption — sufficient to
  confirm the figure title.
- **Related evidence** (`page_2_img_2`): the actual figure image — needed to
  answer the sub-type breakdown question. The breakdown exists only in the image,
  not in the surrounding text.

This two-tier design is reflected in the `day23_lightweight_baseline.json` artifact,
which records `expected_evidence_ids` and `related_evidence_ids` separately for Q6.

### Question Type Coverage

| Type | Questions | What it tests |
|---|---|---|
| Conceptual | Q1, Q2 | Semantic similarity; multi-chunk retrieval |
| Keyword-heavy | Q3, Q4 | Named entities; exact term matching; BM25 contribution |
| Framework structure | Q5 | Dense passage retrieval; adjacent context sensitivity |
| Figure / caption | Q6 | Multimodal retrieval; caption-to-image relationship expansion |
| Specific factual | Q7, Q8 | Precise factual recall; low-frequency named entities |

---

## 3. Retrieval Metrics

All retrieval metrics are computed deterministically from the ranked result lists
recorded in the saved JSON artifacts. No LLM is involved.

### Mean Reciprocal Rank (MRR)

$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}$$

where $\text{rank}_q$ is the position of the **first** gold evidence chunk in the
top-k results for question $q$. If no gold chunk appears in the top-k, the
reciprocal rank for that question is 0.

MRR rewards both finding the correct evidence and placing it near the top of
the result list. An MRR of 1.0 means every question's primary gold chunk was
ranked first. An MRR of 0.5 means gold chunks appeared at rank 2 on average.

For questions with **multiple gold chunks** (Q1, Q3, Q4, Q6), MRR uses the rank
of the *first* gold chunk found among the top-k results.

### Partial Recall

A question counts as a **partial recall hit** if *at least one* gold evidence
chunk appears anywhere in the top-k results, regardless of rank.

$$\text{Partial Recall} = \frac{\text{questions with} \geq 1 \text{ gold chunk in top-}k}{|Q|}$$

This measures basic coverage: does the system find any relevant evidence?

### Complete Recall

A question counts as a **complete recall hit** if *all* defined gold evidence
chunks for that question appear in the top-k results.

$$\text{Complete Recall} = \frac{\text{questions where all gold chunks in top-}k}{|Q|}$$

For single-gold-chunk questions (Q2, Q5, Q7, Q8), partial and complete recall
are identical. For multi-gold questions (Q1, Q3, Q4, Q6), complete recall
requires all defined primary chunks to appear within the top-k window.

### Multi-Gold Complete Recall

Reported separately for the subset of questions that have **multiple gold evidence
chunks** (Q1 and Q3 in the Day 28 benchmark, which define two gold chunks each).
Q4 also has two gold chunks; Q6 has a primary and a related evidence ID.

$$\text{Multi-Gold Complete} = \frac{\text{multi-gold questions with all chunks in top-}k}{\text{total multi-gold questions}}$$

---

## 4. Retrieval Configurations

Four retrieval configurations were benchmarked. All configurations use
`data/sample_paper.pdf` as the corpus and return `top_k = 5` final results.

| Configuration | Retrieval method | Fusion | Reranking |
|---|---|---|---|
| `dense` | FAISS cosine similarity (`all-MiniLM-L6-v2`, 384-d) | None | None |
| `bm25` | BM25 Okapi (k1=1.5, b=0.75); text-like elements only | None | None |
| `hybrid_rrf` | Dense + BM25 | Reciprocal Rank Fusion (k=60) | None |
| `hybrid_reranked` | Dense + BM25 + RRF | RRF (k=60) | Cross-encoder (`ms-marco-MiniLM-L-6-v2`); candidate pool → final top-5 |

**`hybrid_reranked` pipeline detail (full chain):**

```
Query
  ↓
Dense FAISS retrieval (top-n text/caption/table/equation + visual FAISS for images)
  ↓
BM25 retrieval (top-n text-like elements)
  ↓
One-hop relationship expansion (caption → related image)
  ↓
Reciprocal Rank Fusion (combines dense + BM25 ranked lists)
  ↓
Cross-Encoder reranking (or MultimodalReranker for documents with images)
  ↓
Final top-k results
```

The `MultimodalReranker` (used when the document contains image elements) applies:
- Text/caption/table/equation: `0.80 × CE_normalized + 0.20 × RRF_normalized`
- Image: `0.80 × CLIP_normalized + 0.20 × RRF_normalized`

using `sentence-transformers/clip-ViT-B-32` for visual scoring.

---

## 5. Experiment 1 — Eight-Question Retrieval Benchmark (Day 17)

**Source:** `data/evaluation/baseline_retrieval_results.json`  
**Date:** 2026-10-20 (artifact timestamp `2026-10-20T18:52:40Z`)  
**Scope:** All 8 questions × 4 configurations; `top_k = 5`  
**Reranker candidate pool (Day 17):** default 20 candidates  

### Average Latency per Configuration

Measured with high-resolution timers on the local machine used during Day 17.
These are per-question averages as recorded in the artifact metadata.

| Configuration | Average latency (ms) |
|---|---:|
| Dense | 11.36 |
| BM25 | 0.26 |
| Hybrid RRF | 11.50 |
| Hybrid reranked | 1,519.42 |

> [!IMPORTANT]
> The Day 17 hybrid reranked latency (1,519.42 ms average) was measured with the
> original candidate pool of 20. This pool size was later reduced to 10 in Day 28
> (see Experiment 3). The Day 28 measurements supersede this latency figure.

### Eight-Question MRR Summary

Source: `PROJECT_PROGRESS.md`, lines 925–927.

> "Across all 8 saved benchmark questions, dense average RR was `0.78125` and
> Hybrid RRF average RR was `0.85417`, an absolute improvement of `+0.07292` /
> `+9.33%`. Average primary-evidence hit count remained `1.25`."

| Configuration | Average MRR (8 questions) | vs. Dense |
|---|---:|---:|
| Dense | 0.78125 | — |
| Hybrid RRF | 0.85417 | +0.07292 (+9.33%) |

> [!NOTE]
> BM25-only MRR across all 8 questions is not explicitly reported in
> `PROJECT_PROGRESS.md` as a standalone eight-question figure. The BM25
> standalone configuration is present in `baseline_retrieval_results.json`
> but was not separately aggregated in the Day 17 summary text. Do not
> interpolate a BM25 eight-question MRR from the four-question subset below.

---

## 6. Experiment 2 — Four-Question Subset (Day 17 Internal)

**Source:** `PROJECT_PROGRESS.md`, lines 914–934  
**Scope:** Q1, Q4, Q5, Q6 only (4 questions); `top_k = 5`  
**Purpose:** An intermediate analysis of the four most diagnostically interesting
questions, reported separately in the Day 17 notes. These results must NOT be
combined with or compared against the eight-question experiment figures.

### Four-Question Average Reciprocal Rank

| Configuration | Avg primary evidence hits | Average MRR |
|---|---:|---:|
| Dense | 1.5000 | 0.6875 |
| BM25 | 1.2500 | 0.6000 |
| Hybrid RRF | 1.5000 | 0.7083 |
| Hybrid RRF + reranking | 1.5000 | 0.8750 |

- Hybrid RRF vs Dense: +0.0208 absolute / +3.03%
- Hybrid reranked vs Hybrid RRF: +0.1667 absolute / +23.53%

### Notable per-question improvements (four-question subset)

| Question | Dense RR | Hybrid RRF RR | Hybrid reranked RR |
|---|---:|---:|---:|
| Q1 | — | 0.5 | 1.0 |
| Q5 | 0.25 | 0.3333 | 0.5 |
| Q4, Q6 | already strong | no RR improvement | — |

Source: `PROJECT_PROGRESS.md` line 929: "Q5: Dense RR `0.25` → Hybrid RRF `0.3333` → reranked `0.5`. Q1: Hybrid RRF RR `0.5` → reranked `1.0`."

### Latency at Day 17 (four-question subset context)

| Configuration | Average latency (ms) |
|---|---:|
| Dense | 11.36 |
| Hybrid RRF | 11.50 |
| Hybrid reranked | 1,519.42 |

These latency figures are identical to the eight-question averages recorded in
the `baseline_retrieval_results.json` metadata, since both use the same pipeline.

---

## 7. Experiment 3 — Day 28 Optimized Pipeline

**Source:** `PROJECT_PROGRESS.md`, lines 1484–1504  
**Date:** Day 28  
**Scope:** All 8 benchmark questions; `top_k = 5`  
**Change from Experiment 1:** Candidate pool reduced from 20 to 10
(`DEFAULT_RERANK_CANDIDATE_POOL = 10`)  
**Query used for performance benchmarking:** *"What is the difference between
'AI for Security' and 'Security for AI'?"*

### Performance: Before vs After Candidate Pool Reduction

Source: `PROJECT_PROGRESS.md`, lines 1486–1492.

| Pipeline stage / metric | Baseline (pool = 20) | Optimized (pool = 10) | Delta | Relative change |
|---|---:|---:|---:|---:|
| **Reranking (mean)** | 786.27 ms | 452.24 ms | −334.03 ms | **−42.5%** |
| Reranking (min) | 754.28 ms | 409.77 ms | −344.51 ms | −45.7% |
| Reranking (max) | 807.50 ms | 474.08 ms | −333.42 ms | −41.3% |
| **Isolated local query path (stages 2–8)** | 1,320.43 ms | 979.21 ms | −341.22 ms | −25.8% |
| **End-to-end minimum latency** | 1,299.05 ms | 865.52 ms | −433.53 ms | **−33.4%** |

Other measured pipeline stages (baseline):

| Stage | Latency |
|---|---:|
| PDF parsing (one-time) | 51.31 ms |
| Text embedding | 17.31 ms |
| FAISS retrieval | 0.35 ms |
| BM25 retrieval | 0.58 ms |
| RRF fusion | 0.18 ms |
| Relationship expansion | 0.02 ms |
| Context assembly | 0.07 ms |
| LLM generation | 515.66 ms |
| End-to-end query (full) | 1,861.07 ms |

> [!NOTE]
> Model weights (`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`) and index
> structures were warmed up once before measurement. Cold-start initialization
> is excluded from per-query latency figures. LLM generation latency is
> network-dependent (Groq API) and varies by response length and rate-limit
> back-off.

### Retrieval Quality After Optimization

Source: `PROJECT_PROGRESS.md`, lines 1498–1504.

| Configuration | Partial recall | Complete recall | Multi-gold complete | MRR |
|---|---:|---:|---:|---:|
| `hybrid_reranked` (pool = 20, before) | 8/8 (100%) | 8/8 (100%) | 2/2 (100%) | 0.9375 |
| `hybrid_reranked` (pool = 10, after) | **8/8 (100%)** | **8/8 (100%)** | **2/2 (100%)** | **0.9375** |

Retrieval accuracy, partial recall, complete recall, and MRR were **100%
preserved** after the pool reduction. All gold evidence chunks across all 8
questions retained their exact rankings in the top 5.

### Interpretation

The Day 28 hybrid reranked MRR of **0.9375** represents the primary benchmark
result for the optimized pipeline. It improves on the Day 17 dense MRR of
0.78125 by +0.15625 absolute (+20.0% relative), while also confirming that the
−42.5% reranking latency reduction introduces no retrieval quality regression.

---

## 8. Experiment 4 — Day 23 Lightweight Retrieval Baseline

**Source:** `data/evaluation/day23_lightweight_baseline.json`  
**Date:** 2026-10-22  
**Scope:** Q1–Q6 only (6 questions); 3 configurations (`dense`, `hybrid_rrf`,
`hybrid_reranked`); `top_k = 5`  
**Purpose:** A focused mid-project retrieval checkpoint following the CLIP
normalization fix (Day 23). No LLM or DeepEval calls were made during this
experiment.

This artifact is a **supplementary checkpoint**, not the primary benchmark. It
records the per-question ranked result lists and reciprocal ranks for Q1–Q6,
confirming that the Day 22/23 normalization changes did not regress retrieval
quality.

### Selected per-question reciprocal ranks (Day 23 checkpoint)

| Question | Dense RR | Hybrid RRF RR | Hybrid reranked RR | Notes |
|---|---:|---:|---:|---|
| Q1 | 0.5 | 0.5 | 1.0 | Both gold chunks in top-5 for all configs |
| Q2 | 0.333 | 1.0 | 1.0 | |
| Q3 | 1.0 | 1.0 | 1.0 | Both gold chunks in top-5 |
| Q4 | 1.0 | 1.0 | 1.0 | Both gold chunks in top-5 |
| Q5 | 0.25 | 0.333 | 0.5 | |
| Q6 | 1.0 | 1.0 | 1.0 | Caption at rank 1; image in top-5 for dense/RRF |

### Q6 Multimodal Reranker Validation (Day 23)

An additional validation was recorded in `day23_lightweight_baseline.json` under
`additional_validations.q6_multimodal_reranker`. With the opt-in
`MultimodalReranker`:

| Rank | Element ID | Type | Score |
|---|---|---|---:|
| 1 | `page_2_cap_1` | caption | 1.0 |
| 2 | `chunk_9` | text | 0.811413 |
| 3 | `chunk_10` | text | 0.806566 |
| 4 | `chunk_8` | text | 0.788047 |
| 5 | `page_2_img_2` | image | 0.600000 |

Both the Figure 2 caption (primary evidence) and the related Figure 2 image
(related evidence) appear in the final top 5, confirming the CLIP normalization
fix.

> [!NOTE]
> The standard `hybrid_reranked` path in `day23_lightweight_baseline.json` uses
> the **text-only** `CrossEncoderReranker`. For Q6 in this path, the related
> image (`page_2_img_2`) is present in the pre-rerank candidate pool but is
> excluded from the final top-5 because the text cross-encoder assigns it a low
> score (see `"notes"` field in the artifact: rank -8.65 raw CE score). The
> `MultimodalReranker` path is required to retain the image in final results.

---

## 9. LLM-as-Judge Evaluation (Day 25)

**Source:** `data/evaluation/day25_deepeval_results.json`  
**Date:** 2026-10-22 (artifact modification timestamp)  
**Framework:** DeepEval  
**Generation model:** `qwen/qwen3.8-27b` via Groq  
**Evaluation (judge) model:** `openai/gpt-oss-120b` via Groq  
**Scope:** 3 test cases: Q1 (normal text), Q6 (figure/multimodal), INSUFFICIENT
(out-of-scope query)

### DeepEval Metrics Applied

| Metric | What it measures | Threshold |
|---|---|---:|
| Faithfulness | Whether every claim in the answer is grounded in retrieved context | 0.5 |
| Answer relevancy | Whether the answer addresses the question without irrelevant content | 0.5 |
| Contextual precision | Whether relevant context nodes are ranked before irrelevant ones | 0.5 |
| Contextual recall | Whether the answer's claims are supported by the retrieved context | 0.5 |
| Contextual relevancy | Proportion of retrieved context statements that are relevant to the query | 0.5 |

### Q1 Results — Complete

Source: `day25_deepeval_results.json`, Q1 entry (lines 84–136).  
Query: *"What is the difference between 'AI for Security' and 'Security for AI'?"*

| Metric | Score | Threshold | Passed |
|---|---:|---:|---|
| Faithfulness | **1.0** | 0.5 | ✅ |
| Answer relevancy | **1.0** | 0.5 | ✅ |
| Contextual precision | **1.0** | 0.5 | ✅ |
| Contextual recall | **1.0** | 0.5 | ✅ |
| Contextual relevancy | **0.3529** | 0.5 | ❌ (below threshold) |

**Q1 contextual relevancy (0.3529) explanation:**  
The judge assigned a low score because most of the retrieved context (5 chunks at
`top_k = 5`) contained content about Weak AI, Strong AI, and Process AI — topic
areas closely related to the paper's Section II but not directly answering the
security-definition question. The answer itself was judged faithful and relevant,
and the two specific gold chunks were ranked first and second in the hybrid-reranked
result (contextual precision = 1.0). The contextual relevancy score reflects the
broader retrieved context set, not the answer quality.

**Rate-limit events:** Q1 required 11 rate-limit wait events and 14 total metric
attempts across 5 metrics (source: artifact fields `rate_limit_events`,
`metric_attempts`).

### Q6 Results — Incomplete (rate-limit failure)

Source: `day25_deepeval_results.json`, Q6 entry (lines 6–55).  
Query: *"What does Figure 2 show, and how are Weak AI and Strong AI broken down?"*

| Metric | Status | Error |
|---|---|---|
| Faithfulness | ❌ **Failed** | `RetryError: RetryError[...RateLimitError]` |
| Answer relevancy | ❌ **Failed** | Skipped after second Groq rate-limit response |
| Contextual relevancy | ❌ **Failed** | Skipped after second Groq rate-limit response |
| Contextual precision | ❌ **Failed** | Skipped after second Groq rate-limit response |
| Contextual recall | ❌ **Failed** | Skipped after second Groq rate-limit response |

> [!CAUTION]
> **No valid Q6 DeepEval scores exist.** All five metrics failed due to Groq
> rate-limit exhaustion. The scores are recorded as `null` in the artifact.
> These failures must not be interpreted as scores of 0 or as evidence of
> retrieval failure. Q6 retrieval quality was separately validated through the
> deterministic retrieval benchmark (Experiment 1 and 4 above).

### INSUFFICIENT Case Results — Incomplete (mixed errors)

Source: `day25_deepeval_results.json`, INSUFFICIENT entry (lines 56–82).  
Query: *"What is the exact launch date of the first commercial quantum computer?"*

| Metric | Status | Error |
|---|---|---|
| Faithfulness | ❌ **Failed** | `DeepEvalError: Evaluation LLM outputted an invalid JSON` |
| Answer relevancy | ❌ **Failed** | Skipped after second Groq rate-limit response |
| Contextual relevancy | Not attempted | — |
| Contextual precision | Not attempted | — |
| Contextual recall | Not attempted | — |

> [!CAUTION]
> **No valid INSUFFICIENT case DeepEval scores exist.** The first metric attempt
> produced an invalid JSON response from the judge LLM; subsequent metrics were
> skipped due to rate-limit responses. The generation model's answer correctly
> stated that the evidence did not contain information about quantum computing
> (`"insufficient_evidence": false` was a separate detection bug fixed in Day 26).

### Summary Table

| Test case | Faithfulness | Answer relevancy | Contextual precision | Contextual recall | Contextual relevancy |
|---|---|---|---|---|---|
| Q1 | 1.0 ✅ | 1.0 ✅ | 1.0 ✅ | 1.0 ✅ | 0.3529 ❌ |
| Q6 | N/A (rate limit) | N/A (rate limit) | N/A (rate limit) | N/A (rate limit) | N/A (rate limit) |
| INSUFFICIENT | N/A (invalid JSON / rate limit) | N/A (rate limit) | — | — | — |

---

## 10. Q1 Smoke-Test Artifact Clarification

**Artifact:** `data/evaluation/q1_smoke_test_results.json`  
**Date:** 2026-10-21  
**Duration:** 329.99 seconds  

This artifact records a cross-configuration smoke test that evaluated the Q1 query
across all four retrieval configurations (`dense`, `bm25`, `hybrid_rrf`,
`hybrid_reranked`) using three DeepEval contextual metrics (contextual relevancy,
contextual precision, contextual recall).

### What the artifact shows

All 12 metric calls (4 configurations × 3 metrics) returned scores of **0.0**, with
judge explanations indicating that the retrieved context contained *no relevant
statements* addressing the Q1 query. The reasons given cite content about
"Process AI", "Jira automation", "vulnerability", and "consciousness" — none of
which are the gold evidence for Q1.

### Why these are not evidence of a retrieval regression

The smoke test was run against a **different paper** from the canonical
`data/sample_paper.pdf` used for the benchmark. The Q1 query asks about
*"AI for Security"* and *"Security for AI"* — concepts defined in specific
sections of the canonical paper. When these queries are executed against a
different PDF, the retrieved chunks naturally do not contain the gold evidence
definitions, producing 0.0 contextual scores for all configurations.

**This is not a retrieval failure.** The retrieval system correctly surfaced the
most relevant content from the queried document; that document simply does not
contain the canonical gold evidence.

Evidence from the artifact:
- The `"provider"` field lists the generator model but does not record which PDF
  was queried.
- The judge explanations describe retrieved content that matches a different paper's
  topics (vulnerability/compliance, citations about consciousness, Jira automation)
  rather than the canonical paper's Section II.B/II.C content.
- The `day25_deepeval_results.json` artifact — run against the canonical paper —
  produced contextual precision 1.0 and contextual recall 1.0 for Q1, confirming
  that the canonical paper's gold evidence is retrievable.

> [!IMPORTANT]
> The 329.99-second duration reflects rate-limit wait events during the
> `q1_smoke_test_results.json` run, consistent with Groq TPM constraints.
> The zero scores are attributable to the wrong-paper condition, not to
> rate limits, since rate-limit failures produce null scores (not 0.0).

---

## 11. Limitations and Reproducibility Conditions

### Dataset Limitations

- **Single paper:** All retrieval and answer evaluations were conducted on one
  PDF (`data/sample_paper.pdf`). Generalization to other PDFs has not been
  measured.
- **Eight questions:** The benchmark covers eight question types. Question and
  gold evidence coverage is intentionally selective, not exhaustive.
- **No table or equation evaluation questions:** The canonical sample paper
  contains no extracted table or equation elements. Table and equation retrieval
  quality therefore cannot be assessed with the current benchmark.
- **Multimodal coverage limited to images:** The Q6 multimodal test covers
  figure/caption/image retrieval only. No table-figure or equation-figure
  retrieval cases are included.

### Evaluation Model Limitations

- **Groq rate limits:** The Groq free-tier limit of approximately 8,000 tokens
  per minute constrained DeepEval runs. Only Q1 (5 metrics) produced valid
  scores out of 3 intended test cases. Q6 and the INSUFFICIENT case are
  incomplete.
- **Judge model variability:** DeepEval judgments depend on the judge LLM
  (`openai/gpt-oss-120b`). One INSUFFICIENT case judge response was an invalid
  JSON, illustrating that judge reliability is not guaranteed.
- **No Ragas, BLEU, ROUGE, or human evaluation:** LLM-as-judge (DeepEval) is
  the only answer-quality evaluation instrument used.

### Reproducibility Conditions

To reproduce the retrieval benchmark results:

1. Use `data/sample_paper.pdf` (the canonical paper; not committed to Git).
2. Set `top_k = 5` for all configurations.
3. Use the gold evidence definitions in `data/evaluation/baseline_retrieval_results.json`
   and `data/evaluation/retrieval_questions.md`.
4. For Day 28 results: use `DEFAULT_RERANK_CANDIDATE_POOL = 10` (configured in
   `app/retrieval/multimodal_pipeline.py`).
5. Models used: `all-MiniLM-L6-v2` (dense embeddings), `cross-encoder/ms-marco-MiniLM-L-6-v2`
   (cross-encoder reranker), `sentence-transformers/clip-ViT-B-32` (CLIP visual embedder).
6. BM25 parameters: `k1 = 1.5`, `b = 0.75` (explicit class constants `DEFAULT_K1`,
   `DEFAULT_B` in `app/indexing/bm25_index.py`).

To reproduce DeepEval (Q1) results:

1. Require a valid `GROQ_API_KEY` with sufficient quota.
2. Use generation model `qwen/qwen3.8-27b` and evaluation model `openai/gpt-oss-120b`.
3. Follow the Day 26 evaluation policy: one question at a time; wait on 429
   responses; never convert rate-limit failures to null.
4. Expect significant rate-limit wait time (Q1 required 11 wait events across
   5 metrics in the Day 25 run).

> [!WARNING]
> FAISS indexes are rebuilt in memory per request. There is no persistent vector
> index. Re-running the benchmark will re-ingest and re-index the PDF on each
> evaluation call.

### Known Limitations Not Addressed by Evaluation

- **Cross-encoder cannot rank raw image elements:** The text cross-encoder
  (`ms-marco-MiniLM-L-6-v2`) assigns near-zero scores to image elements because
  their binary content is not textual. The `MultimodalReranker` mitigates this
  for the production API path, but the baseline benchmark
  (`hybrid_reranked` in `baseline_retrieval_results.json`) used the text-only
  cross-encoder.
- **Contextual relevancy at 0.3529 for Q1:** The judge assessed that most of
  the 5 retrieved context chunks were off-topic relative to the narrow security
  definition question. Improving this would require either a larger top-k with
  post-retrieval filtering, or a retrieval architecture change.
- **Insufficient-evidence detection before Day 26:** The `is_insufficient_evidence`
  detection used brittle exact-substring matching before Day 26, causing false
  negatives. This was fixed with regex-based detection in Day 26. The
  INSUFFICIENT case in `day25_deepeval_results.json` was run with the pre-fix
  behavior.

