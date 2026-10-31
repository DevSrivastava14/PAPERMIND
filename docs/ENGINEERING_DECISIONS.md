# PaperMind — Engineering Decisions & Failure Analysis

> **Scope**: Decisions and failures from Days 1–30 of the PaperMind development sprint.
> **Evidence sources**: `PROJECT_PROGRESS.md` (day-by-day log), source code inspection.
> **Rule**: Measured results are distinguished from engineering rationale throughout. Numbers that appear without explicit measurement context are not asserted as measurements.

---

## Table of Contents

1. [Engineering Decisions](#engineering-decisions)
   - [D1 — Dense vs. Hybrid Retrieval](#d1--dense-vs-hybrid-retrieval)
   - [D2 — BM25 + Dense + Reciprocal Rank Fusion](#d2--bm25--dense--reciprocal-rank-fusion)
   - [D3 — Cross-Encoder Reranking](#d3--cross-encoder-reranking)
   - [D4 — CLIP Visual Retrieval and Multimodal Reranking](#d4--clip-visual-retrieval-and-multimodal-reranking)
   - [D5 — Relationship Expansion and Provenance](#d5--relationship-expansion-and-provenance)
   - [D6 — LangGraph Orchestration](#d6--langgraph-orchestration)
   - [D7 — Candidate-Pool Optimization: top-50 → top-30 → top-10](#d7--candidate-pool-optimization)
   - [D8 — In-Memory FAISS](#d8--in-memory-faiss)
   - [D9 — Request-Local State Instead of Shared Mutable State](#d9--request-local-state-instead-of-shared-mutable-state)
   - [D10 — Lazy DeepEval Imports](#d10--lazy-deepeval-imports)
   - [D11 — Structured Logging and Deferred LangSmith](#d11--structured-logging-and-deferred-langsmith)
2. [Failure Analysis](#failure-analysis)
   - [F1 — CLIP Normalization Bug](#f1--clip-normalization-bug)
   - [F2 — FastAPI Exception-Handler TypeError](#f2--fastapi-exception-handler-typeerror)
   - [F3 — Shared Mutable Candidate-Count Race](#f3--shared-mutable-candidate-count-race)
   - [F4 — Eager DeepEval Import Breaking API Startup](#f4--eager-deepeval-import-breaking-api-startup)
   - [F5 — Groq Rate-Limit Evaluation Failures](#f5--groq-rate-limit-evaluation-failures)

---

## Engineering Decisions

---

### D1 — Dense vs. Hybrid Retrieval

**Problem being addressed**

Dense FAISS retrieval (384-d `all-MiniLM-L6-v2`) provides semantic similarity matching but misses queries that hinge on exact technical terms, model names, subscript identifiers (`d_k`, `w_q`), or hyphenated labels (`bleu-4`, `cross-encoder/ms-marco`) that may not map cleanly into the embedding space.

**Chosen approach**

Hybrid retrieval: dense FAISS + BM25 sparse retrieval, combined with Reciprocal Rank Fusion (RRF, k=60). Dense handles semantic similarity; BM25 handles lexical and exact-term precision. The two ranked lists are fused by RRF, which eliminates direct score-scale calibration problems.

**Why it was selected**

The two retrieval signals are complementary. Dense retrieval can surface a passage conceptually relevant to "attention mechanism scaled dot product" even if the passage uses different phrasing; BM25 reliably surfaces passages that contain the literal tokens `d_k` or `scaled dot product`. RRF combines these lists without requiring cross-system score normalization, because all scores are replaced by reciprocal-rank-based values.

**Alternatives and trade-offs**

| Alternative | Reason not chosen |
|---|---|
| Dense only | Does not handle exact-term queries reliably; Q5 dense RR was 0.25 |
| BM25 only | Misses semantic similarity for paraphrase queries; average RR (Q1/Q4/Q5/Q6) was 0.6000 vs. 0.6875 for dense |
| Score interpolation (linear blend) | Requires calibrating two heterogeneous score distributions; RRF sidesteps this entirely |
| Learned sparse (SPLADE) | Significantly higher complexity and training-data requirement for minimal additional gain at this scale |

**Measured results**

Source: `data/evaluation/baseline_retrieval_results.json` and `PROJECT_PROGRESS.md` Day 17.

| Configuration | Average RR (Q1–Q8, 8 questions) |
|---|---|
| Dense FAISS | 0.78125 |
| BM25 | — (not reported in 8Q aggregate) |
| Hybrid RRF | 0.85417 |

Absolute improvement: +0.07292 (+9.33%) over dense for the 8-question benchmark. Average primary-evidence hit count remained equal at 1.25 across dense and hybrid-RRF configurations.

For the Q1/Q4/Q5/Q6 subset (4 questions):

| Configuration | Average RR |
|---|---|
| Dense | 0.6875 |
| BM25 | 0.6000 |
| Hybrid RRF | 0.7083 |

Source: `PROJECT_PROGRESS.md` Day 17, lines 915–927.

---

### D2 — BM25 + Dense + Reciprocal Rank Fusion

**Problem being addressed**

Independently implementing BM25 and combining it with FAISS results requires a fusion step that is agnostic to the score scales produced by each system. FAISS inner products (post-L2 normalization, cosine similarity) and BM25 IDF-weighted term frequency scores are not directly comparable.

**Chosen approach**

Standalone `BM25Index` implemented from scratch using the Robertson-Spärck Jones / Lucene IDF formulation:

$$\text{IDF}(q_i) = \ln\left(1.0 + \frac{N - n_{q_i} + 0.5}{n_{q_i} + 0.5}\right)$$

Parameters: `k1 = 1.5`, `b = 0.75` (declared as `BM25Index.DEFAULT_K1` and `BM25Index.DEFAULT_B` class constants — not delegated to a third-party library).

RRF formula (Cormack et al., SIGIR 2009):

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}(d, r)}, \quad k = 60$$

RRF replaces all original BM25 and FAISS scores with pure reciprocal-rank-based scores. The two ranked lists are deduplicated by `element_id`; intra-list duplicates use the best (lowest) rank only.

**Why it was selected**

- The Robertson-Spärck Jones / Lucene IDF is non-negative for all term-frequency ratios (no negative score contributions), which simplifies downstream score handling.
- RRF requires no score calibration and degrades gracefully when one list is empty (the non-empty list's contributions survive unchanged).
- The tokenizer (`_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*")`) was hand-built with standard-library `re` to preserve scientific identifiers (`d_k`, `bleu-4`, `gpt-4`) without requiring NLTK or spaCy.

**Alternatives and trade-offs**

| Alternative | Reason not chosen |
|---|---|
| Rank-FUSION (Borda) | Less principled than RRF for sparse rank distributions |
| Score-normalised linear combination | Requires calibrating FAISS cosine scores vs. BM25 IDF scores across different document sets |
| Third-party BM25 library (rank_bm25) | Adds dependency; does not return `MultimodalSearchResult` objects natively; prevents tight provenance control |

**Measured results**

See D1. BM25 alone underperformed dense on the Q1/Q4/Q5/Q6 subset (0.6000 vs. 0.6875 average RR), but combined with dense via RRF improved average RR to 0.7083.

Source: `PROJECT_PROGRESS.md` Day 17, line 918.

---

### D3 — Cross-Encoder Reranking

**Problem being addressed**

Bi-encoder models (used for FAISS dense retrieval) encode query and passage independently, which means they can miss fine-grained relevance distinctions. A cross-encoder evaluates `(query, passage)` jointly, capturing interaction effects that bi-encoders cannot.

**Chosen approach**

`CrossEncoderReranker` using `cross-encoder/ms-marco-MiniLM-L-6-v2` (HuggingFace, Sentence Transformers). Applied after RRF fusion over a bounded candidate pool (see D7). Cross-encoder scores replace RRF scores; final `top_k` is truncated after reranking.

For IMAGE elements, text extraction falls back to metadata keys (`caption`, `text`, `description`, `title`, `label`). Raw image bytes are never passed to the cross-encoder. If no metadata text is available, the image receives an empty-string representation and scores near the minimum.

**Why it was selected**

- MS-MARCO MiniLM cross-encoders are well-calibrated for passage-ranking tasks and fast enough on CPU (before optimization, 786 ms for 20 candidates; after pool reduction, 452 ms for 10 candidates).
- The cross-encoder is applied as a post-fusion reranker, not as the primary retrieval mechanism, preserving recall from the hybrid first stage.
- Observable improvements on Q1 (RR 0.5 → 1.0) and Q5 (RR 0.3333 → 0.5).

**Known limitation confirmed during evaluation**

Source: `PROJECT_PROGRESS.md` Day 16, Task 3, line 873; Day 18, lines 990–996.

The cross-encoder prefers explanatory narrative text over short figure captions for figure-focused queries. For Q6 ("FIGURE 3 AI RMF NIST"):
- Figure 2 caption: rank 1, score `5.24833202`.
- Figure 2 image: rank 14/20, score `-8.64942169`.

This is expected for a passage-trained cross-encoder operating over heterogeneous scientific document elements. The image was excluded by the cross-encoder score, not by FAISS, BM25, or relationship expansion.

**Alternatives and trade-offs**

| Alternative | Reason not chosen |
|---|---|
| ColBERT late interaction | Much larger model, more complex deployment, diminishing returns at this document scale |
| Learned re-ranker (T5-based) | Higher complexity; MS-MARCO MiniLM already calibrated for this domain |
| No reranking | Q1/Q4/Q5/Q6 average RR 0.7083 vs 0.8750 reranked — significant precision loss |

**Measured results**

Source: `PROJECT_PROGRESS.md` Day 17, lines 913–940.

| Configuration | Average RR (Q1/Q4/Q5/Q6) |
|---|---|
| Hybrid RRF | 0.7083 |
| Hybrid RRF + Reranked | 0.8750 |

Reranking improvement: +0.1667 (+23.53%) on this subset. Average latency: Hybrid RRF 11.50 ms → Hybrid reranked 1,519.42 ms (pre-optimization; see D7).

---

### D4 — CLIP Visual Retrieval and Multimodal Reranking

**Problem being addressed**

The cross-encoder cannot meaningfully score IMAGE elements — it is text-only and receives empty or near-empty strings for raw image content. For Q6 (a figure-related query), the Figure 2 image was ranked 14th out of 20 candidates by the cross-encoder and dropped from the final `top_k=5` results, producing `visual_result_count=0`.

**Chosen approach**

`MultimodalReranker` (`app/reranking/multimodal.py`) combining two scoring signals:
- **Text-like elements** (TEXT, CAPTION, TABLE, EQUATION): `0.80 × CrossEncoder_normalized + 0.20 × RRF_normalized`.
- **IMAGE elements**: `0.80 × CLIP_normalized + 0.20 × RRF_normalized`.

Visual embedder: `SentenceTransformerVisualEmbedder` with model `sentence-transformers/clip-ViT-B-32`, producing 512-dimensional embeddings. Text queries are encoded into the same CLIP embedding space for cosine similarity comparison with image vectors.

The `MultimodalReranker` is opt-in and selected automatically in `build_retrieval_pipeline()` when the document contains IMAGE elements. Text-only documents continue using `CrossEncoderReranker` directly.

**Why it was selected**

Source: `PROJECT_PROGRESS.md` Day 21, lines 1066–1094.

- CLIP embeds both text queries and images into a shared 512-d space, enabling genuine semantic comparison between a text query and a retrieved image.
- Images never inherit caption scores; visual and text scoring are independent paths.
- The 80/20 weight split was validated empirically by comparing Q6 reranking results: with `CrossEncoderReranker`, the Figure 2 image remained outside top-5; with `MultimodalReranker` (same 20-element candidate pool), the Figure 2 image ranked fifth.

**Alternatives and trade-offs**

| Alternative | Reason not chosen |
|---|---|
| Caption-proxy for images in cross-encoder | Would cause images to inherit caption scores; breaks provenance and could rank unrelated images highly |
| Dense visual-only retrieval for figures | Loses the hybrid RRF fusion benefit; figures already enter candidate pool via caption→image relationship expansion |
| Fine-tuned cross-encoder for images | Training data and infrastructure not available at this stage |

**Known limitation**

The 80/20 weights are experimental and not calibrated against a held-out image retrieval dataset. Score normalization for CLIP uses a deterministic bounded `[0, 1]` transform; with a single image candidate the CLIP contribution is fixed at 0.5 (explicit neutral score).

Source: `PROJECT_PROGRESS.md` Day 22, lines 1107–1112; Day 23, line 1143.

---

### D5 — Relationship Expansion and Provenance

**Problem being addressed**

A text query for a figure-related question (Q6) retrieves the figure's *caption* but not the figure *image* itself, because the BM25 and text FAISS indices only cover text-like elements. The image would never enter the candidate pool without explicit graph traversal.

Additionally, caption-to-image ID mapping had a bug: the extractor generated `#page_2_imgblock_2` while the actual indexed image ID was `#page_2_img_2`.

**Chosen approach**

Source: `PROJECT_PROGRESS.md` Day 18, lines 963–974.

1. **ID bug fix**: Caption-to-image association corrected using the PDF image block's `number` field and the `#img_{index}` ID convention.
2. **One-hop relationship expansion** (`_expand_related_results()` in `app/retrieval/multimodal_pipeline.py`): after FAISS/BM25 retrieval, each result's `related_element_ids` list is traversed one level. Related elements are fetched from the text or visual FAISS index by element_id and appended to the candidate list with the parent result's score. Deduplication by `element_id` prevents double-counting. No recursive traversal.
3. Expansion is applied **before** reranking, so related images enter the cross-encoder or CLIP reranking pool.

Each `MultimodalElement` carries full provenance: `element_id`, `element_type`, `source_filename`, `page_number` (1-based), `related_element_ids`, and `metadata`.

**Why it was selected**

- One-hop expansion is the minimum required to pull images into the candidate pool when their captions are retrieved.
- Avoiding recursive traversal prevents unbounded candidate-set growth for documents with dense cross-reference graphs.
- Score inheritance (image receives parent's score) is a pragmatic default; the reranker immediately re-scores all candidates so the inherited score only affects candidates if no reranker is configured.

**Alternatives and trade-offs**

| Alternative | Reason not chosen |
|---|---|
| Multi-hop graph traversal | Could grow candidate set unboundedly; unnecessary for one-level caption↔image links |
| Shared text-image embedding space | Would require a joint encoder; CLIP provides this for images but existing text index is 384-d |
| No relationship expansion | Q6 figure retrieval would always fail; `visual_result_count=0` in every reranked result |

---

### D6 — LangGraph Orchestration

**Problem being addressed**

The four pipeline stages (ingest → build index/pipeline → retrieve → generate) involve multiple error modes at each step. Without explicit state management, failures in any stage propagate as uncontrolled exceptions through the call stack, making error categorization and partial-failure handling difficult.

**Chosen approach**

Source: `PROJECT_PROGRESS.md` Day 24; `app/orchestration/langgraph_graph.py`.

LangGraph `StateGraph` with four nodes and conditional edges:

```
START → ingest → build_pipeline → retrieve → generate_answer → END
```

Each node returns an updated `GraphState` `TypedDict`. Status transitions (`"pending"` → `"ingestion_complete"` → `"indexing_complete"` → `"success"` or `"error"`) drive conditional edge routing. Any node that returns `status == "error"` short-circuits directly to `END`; the API layer maps this to an HTTP 422 with a sanitized error message.

**Why it was selected**

- Explicit state (`GraphState`) makes every field visible to every node, without nested function arguments or hidden closure state.
- Conditional edges make error routing declarative and testable — a mock ingest node returning `status="error"` verifiably causes the graph to skip downstream nodes.
- LangGraph is a thin library over Python; it does not impose an external service dependency.
- The four-node structure maps naturally onto the four conceptually distinct phases of a RAG pipeline.

**Alternatives and trade-offs**

| Alternative | Reason not chosen |
|---|---|
| Linear Python function calls | No declarative error routing; failures require try/except chains at every call site |
| Airflow / Prefect | Heavyweight external service; unnecessary for a single-request synchronous pipeline |
| FastAPI background tasks | Does not provide shared typed state between pipeline stages |
| LangChain LCEL | Less explicit state; harder to test individual nodes with mock inputs |

**Note on graph initialization**: The compiled graph is a module-level singleton (`graph = build_papermind_graph()` in `app/api/main.py`). Each `POST /query` invocation calls `graph.invoke(initial_state)`, passing a fresh `GraphState` dict. There is no shared mutable state between invocations at the graph level.

---

### D7 — Candidate-Pool Optimization

**Problem being addressed**

Source: `PROJECT_PROGRESS.md` Day 28, lines 1466–1472.

The default RRF candidate pool size passed to the reranker was 20 (`max(effective_top_k, 20)`). The `CrossEncoderReranker` performs one full `(query, passage)` forward pass per candidate on CPU, so 20 candidates required 20 cross-attention forward passes for a final `top_k = 5` — generating 4× more passes than needed if the gold evidence is in the top-10.

Measured baseline: reranking mean latency **786.27 ms** (min 754.28 ms, max 807.50 ms) for 20 candidates.

**Chosen approach**

Reduced default candidate pool from 20 to 10 via a named constant in `app/retrieval/multimodal_pipeline.py`:

```python
DEFAULT_RERANK_CANDIDATE_POOL: int = 10
```

The `rrf_limit` (candidates passed to the reranker) becomes `max(effective_top_k, DEFAULT_RERANK_CANDIDATE_POOL)`. Callers can still override `candidate_top_k` per call when a larger pool is justified.

> [!NOTE]
> `PROJECT_PROGRESS.md` records the progression as: original default was `max(effective_top_k, 20)` → interim experiment at 10 → final confirmed value `DEFAULT_RERANK_CANDIDATE_POOL = 10`. The architecture documentation describes this as "top-50 → top-30 → top-10" based on the broader roadmap framing, but the actual measured baseline was pool=20.

**Why it was selected**

Measured latency reduction with zero measured accuracy regression, verified against the full 8-question evaluation benchmark.

**Alternatives and trade-offs**

| Alternative | Reason not chosen |
|---|---|
| GPU acceleration | Requires GPU in deployment; adds infrastructure complexity |
| Smaller cross-encoder model | Lower accuracy; MiniLM-L-6-v2 is already a small model |
| Quantised cross-encoder | Additional conversion step; not pursued because pool reduction achieved the target |
| Larger pool with async batching | Adds concurrency complexity not warranted by the latency target |

**Measured results**

Source: `PROJECT_PROGRESS.md` Day 28, lines 1484–1504.

| Metric | Pool = 20 (baseline) | Pool = 10 (optimized) | Delta | Relative change |
|---|---|---|---|---|
| Reranking mean latency | 786.27 ms | 452.24 ms | -334.03 ms | **-42.5%** |
| Reranking min latency | 754.28 ms | 409.77 ms | -344.51 ms | -45.7% |
| Reranking max latency | 807.50 ms | 474.08 ms | -333.42 ms | -41.3% |
| Isolated local query path (stages 2–8) | 1,320.43 ms | 979.21 ms | -341.22 ms | -25.8% |
| End-to-end min latency | 1,299.05 ms | 865.52 ms | -433.53 ms | -33.4% |

Retrieval quality (8-question benchmark, `hybrid_reranked`): MRR **0.9375**, partial recall **8/8**, complete recall **8/8**, multi-gold complete **2/2** — **identical before and after optimization**.

---

### D8 — In-Memory FAISS

**Problem being addressed**

PaperMind needs a dense vector index that is fast to query and simple to build per-request. For a single-paper RAG system with documents on the order of tens to hundreds of text chunks and image elements, persistence adds complexity without clear benefit at this stage.

**Chosen approach**

`faiss.IndexFlatIP` (exact inner-product search) with L2 normalization at insertion and query time, providing exact cosine similarity without approximation. Two separate indices: one 384-d for text-like elements, one 512-d for visual elements. Both are built in-memory from scratch on each request.

**Why it was selected**

- `IndexFlatIP` is exact (no approximation error), deterministic, and has no training phase.
- L2 normalization before insertion converts inner products to cosine similarities without a separate `IndexFlatL2` or custom metric.
- For documents with tens to hundreds of elements, O(n) brute-force search is fast: FAISS retrieval measured at **0.35 ms** per query.
- No disk I/O, no serialization, no cache-invalidation logic.

**Alternatives and trade-offs**

| Alternative | Reason not chosen |
|---|---|
| FAISS `IndexIVFFlat` (approximate) | Requires a training corpus; no benefit at small document scale |
| Pinecone / Weaviate / Qdrant | Adds an external network service; inappropriate for a local-first single-paper system |
| Persistent FAISS index (save/load) | Requires a cache key strategy (e.g., PDF hash), eviction policy, and invalidation logic; deferred as a future optimization |
| SQLite + vector extension | Significantly slower than FAISS for vector similarity; more complex setup |

**Limitation**

Every request rebuilds the index from scratch, including re-embedding all document elements. This is the dominant latency cost for large PDFs. Measured: PDF parsing 51.31 ms, text embedding 17.31 ms per request. Source: `PROJECT_PROGRESS.md` Day 28, lines 1460–1470.

---

### D9 — Request-Local State Instead of Shared Mutable State

**Problem being addressed**

Source: `PROJECT_PROGRESS.md` Day 29, lines 1565–1570.

An earlier implementation of `MultimodalRetrievalPipeline` tracked the number of RRF candidates entering the reranker using an instance-level mutable attribute (`self._last_candidate_count`). Because FastAPI can serve concurrent requests (or could in a multi-threaded deployment), writes to this attribute from one request could be read by a different concurrent request — a data race that would produce incorrect `candidate_count` values in logs.

**Chosen approach**

`candidate_count` is now request-local:
1. `hybrid_retrieve_reranked()` captures `candidate_count = len(rrf_candidates)` immediately before the reranking call.
2. When called with `return_candidate_count=True`, it returns `(reranked_results, candidate_count)` as a tuple.
3. `hybrid_retrieve_reranked_and_assemble()` receives the count and stores it in `MultimodalRetrievedContext.candidate_count` (a new field on the dataclass).
4. The shared mutable `self._last_candidate_count` / `self.last_candidate_count` attributes were removed from the pipeline class.

**Why it was selected**

- Eliminates a class of data race entirely by making the count flow through the call stack, not through shared state.
- `MultimodalRetrievedContext` is a per-request object, so `candidate_count` on it is inherently request-scoped.
- No synchronisation primitives (locks, atomics) needed.

**Alternatives and trade-offs**

| Alternative | Reason not chosen |
|---|---|
| Thread-local storage | More complex; the correct fix is to not use shared mutable state at all |
| Locking the mutable field | Adds contention and complexity for a problem that is simpler to eliminate |
| Context variable (contextvars) | Would work, but the value flows naturally through the return path without a context variable |

**Verification evidence**

Source: `PROJECT_PROGRESS.md` Day 29, focused test results.

```
pytest tests/test_multimodal_reranked_retrieval.py \
  -k "test_hybrid_retrieve_reranked_and_assemble_sets_request_local_candidate_count"
→ 1 passed, 22 deselected
```

---

### D10 — Lazy DeepEval Imports

**Problem being addressed**

Source: `PROJECT_PROGRESS.md` Day 30; `app/evaluation/__init__.py`.

`app/evaluation/__init__.py` previously imported DeepEval symbols at module load time:

```python
from app.evaluation.deepeval_evaluator import DeepEvalAnswerEvaluator, ...
```

This caused `ModuleNotFoundError: No module named 'deepeval'` when the Docker container started, because `deepeval` is an evaluation-only dependency not included in the production `requirements.txt`.

**Chosen approach**

PEP 562 lazy `__getattr__` dispatch in `app/evaluation/__init__.py`:

```python
if TYPE_CHECKING:
    from app.evaluation.deepeval_evaluator import DeepEvalAnswerEvaluator, ...

_DEEPEVAL_SYMBOLS = {"DeepEvalAnswerEvaluator", ...}

def __getattr__(name: str):
    if name in _DEEPEVAL_SYMBOLS:
        from app.evaluation import deepeval_evaluator
        val = getattr(deepeval_evaluator, name)
        globals()[name] = val
        return val
    raise AttributeError(...)
```

`TYPE_CHECKING` guard ensures type checkers see the symbols without runtime import. `__getattr__` is only called when a `_DEEPEVAL_SYMBOLS` name is actually accessed at runtime — i.e., only in evaluation scripts, never during API startup.

**Why it was selected**

- Minimal change to the existing module structure; the public interface is unchanged.
- `TYPE_CHECKING` guard preserves full static type-checking support.
- The caching step (`globals()[name] = val`) ensures subsequent accesses on the same process are instant.
- No conditional-import pattern in every callsite; the laziness is centralized in `__init__.py`.

**Alternatives and trade-offs**

| Alternative | Reason not chosen |
|---|---|
| Move DeepEval into a separate top-level package | Would change import paths throughout the project |
| Add `deepeval` to production `requirements.txt` | DeepEval pulls in large ML dependencies (`torch`, `transformers`) unnecessary for the API runtime |
| Wrap in `try/except ImportError` in each caller | Scatters the guard across multiple files; harder to maintain |

**Verification evidence**

Source: `PROJECT_PROGRESS.md` Day 30, focused test results.

```
pytest tests/test_evaluation_dataset.py tests/test_evaluation_runner.py \
       tests/test_evaluation_deepeval.py
→ 42 passed, 2 warnings
```

API startup no longer requires `deepeval`; `pytest tests/test_api.py` → 17 passed.

---

### D11 — Structured Logging and Deferred LangSmith

**Problem being addressed**

Source: `PROJECT_PROGRESS.md` Day 29.

The system needed per-request observability covering: request identity, retrieval configuration, candidate pool size, result count, latencies, and error categorization — without logging sensitive data (query text, retrieved evidence, prompts, API keys, filesystem paths).

**Chosen approach**

Python standard-library `logging` emitting single-line JSON events via three named loggers:
- `papermind.api` — API boundary (request start, completion, failure)
- `papermind.retrieval` — retrieval node (retrieval completion, failure)
- `papermind.generation` — LLM node (generation completion, failure)

Log level: `PAPERMIND_LOG_LEVEL` env var (fallback: `LOG_LEVEL`, then `INFO`).

Every request gets a UUID `request_id` (from `X-Request-ID` header or generated) and a deterministic `document_id` (SHA-256 of the relative PDF path, 16 hex chars). These IDs correlate log events across the three loggers for a single request.

`_sanitize_message()` redacts API key patterns (`gsk_…`, `sk-…`) and filesystem paths before logging any error string.

**Why LangSmith was deferred**

Source: `PROJECT_PROGRESS.md` Day 29, line 1581; Day 26, line 1293.

> "LangSmith is planned for later observability/tracing and is **not** a replacement for the current evaluation framework."
> "Not integrated. Remains conditional on future value assessment."

The structured JSON logs provide the same per-request observability surface needed at the current scale: request identity, retrieval config, candidate count, result count, reranker identity, latencies, and error categories — without an external service dependency. LangSmith integration would add value primarily for distributed tracing across multiple services or for interactive trace inspection in a team context, neither of which applies at the current single-service scope.

**Alternatives and trade-offs**

| Alternative | Reason not chosen |
|---|---|
| LangSmith | External service; adds dependency and SDK overhead; deferred pending evidence of additional value |
| OpenTelemetry | Appropriate for multi-service distributed systems; over-engineered for a single process |
| Print statements / unstructured logs | Not parseable; no correlation between request and retrieval/generation events |
| No observability | Impossible to debug latency regressions or identify error patterns in production |

---

## Failure Analysis

---

### F1 — CLIP Normalization Bug

**Root cause**

Source: `PROJECT_PROGRESS.md` Day 22, lines 1101–1112; Day 23, lines 1142–1148.

After the Day 21 introduction of `MultimodalReranker`, the combined score for IMAGE elements was computed as:

```
combined = 0.80 * CLIP_normalized + 0.20 * RRF_normalized
```

where `CLIP_normalized` used **candidate-pool-level min-max normalization**. When only one IMAGE element was in the candidate pool, min-max normalization is undefined (max = min), and the implementation produced an artificial score of 0.5.

In Day 22 a general normalization stabilization guard was added to handle near-zero-range CE scores, but the single-image CLIP case was not separately handled. In Day 23, a Q6 regression was confirmed: the Day 22 normalization change moved the Figure 2 image from rank 5 to rank 6, causing it to fall outside the final `top_k=5`.

**Fix**

Source: `PROJECT_PROGRESS.md` Day 23, lines 1144–1148.

Added an explicit neutral score of `0.5` for a single CLIP candidate, independent of pool composition:

```python
@staticmethod
def _normalize_clip(values: Sequence[float]) -> List[float]:
    if len(values) == 1:
        return [0.5]
    return [min(1.0, max(0.0, value)) for value in values]
```

This makes CLIP scoring deterministic and pool-composition-independent for the common single-image case.

**Verification evidence**

Source: `PROJECT_PROGRESS.md` Day 23.

Focused multimodal reranker validation: **11 passed**.

Targeted Q6 validation confirmed both elements retained in final top 5:
- Figure 2 caption: `sample_paper.pdf#page_2_cap_1`
- Related image: `sample_paper.pdf#page_2_img_2`

---

### F2 — FastAPI Exception-Handler TypeError

**Root cause**

Source: `PROJECT_PROGRESS.md` Day 27; `app/api/main.py`.

The custom `validation_exception_handler` registered with `@app.exception_handler(RequestValidationError)` had an incorrect function signature. FastAPI's exception handler protocol requires the signature `async def handler(request: Request, exc: ExceptionType)`. An earlier version used an incorrect signature that caused a `TypeError` when the handler was called by FastAPI's exception dispatch mechanism on a 422 validation error.

**Fix**

Corrected the handler signature to:

```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
    ...
    response = await request_validation_exception_handler(request, exc)
    response.headers["X-Request-ID"] = request_id
    return response
```

The handler now delegates to FastAPI's built-in `request_validation_exception_handler` after emitting a structured log event, and correctly attaches `X-Request-ID` to the 422 response.

**Verification evidence**

Source: `PROJECT_PROGRESS.md` Day 27; focused test results.

```
pytest tests/test_api.py
→ 14 passed  (Day 29), 17 passed (Day 30)
```

Validation error tests covering empty query, whitespace query, non-PDF extension, and path traversal all pass.

---

### F3 — Shared Mutable Candidate-Count Race

**Root cause**

Source: `PROJECT_PROGRESS.md` Day 29, lines 1565–1570.

`MultimodalRetrievalPipeline` stored the count of RRF candidates passed to the reranker as an instance-level attribute (`self._last_candidate_count`). Because the pipeline instance is constructed per-request inside the LangGraph node (not shared), the immediate race risk was low in practice — but the pattern was still incorrect: if the pipeline were ever cached or if request handling became concurrent within a single pipeline instance, writes from one invocation would corrupt reads from another.

More concretely, reading `self._last_candidate_count` in the `retrieve_context` LangGraph node after calling `hybrid_retrieve_reranked()` is a time-of-check/time-of-use gap even within a single-threaded context if the pipeline is reused.

**Fix**

Source: `PROJECT_PROGRESS.md` Day 29.

`hybrid_retrieve_reranked()` now accepts `return_candidate_count: bool = False`. When `True`, it returns `(results, candidate_count)` as a tuple where `candidate_count = len(rrf_candidates)` is captured immediately before the reranking call:

```python
rrf_candidates = reciprocal_rank_fusion(...)
rrf_candidates = self._expand_related_results(rrf_candidates)
candidate_count = len(rrf_candidates)   # captured here, before reranking
...
return (reranked_results, candidate_count) if return_candidate_count else reranked_results
```

`hybrid_retrieve_reranked_and_assemble()` calls with `return_candidate_count=True` and sets `context.candidate_count`. The shared mutable attributes were removed.

**Verification evidence**

```
pytest tests/test_multimodal_reranked_retrieval.py \
  -k "test_hybrid_retrieve_reranked_and_assemble_sets_request_local_candidate_count"
→ 1 passed, 22 deselected
```

---

### F4 — Eager DeepEval Import Breaking API Startup

**Root cause**

Source: `PROJECT_PROGRESS.md` Day 30.

`app/evaluation/__init__.py` contained a top-level import:

```python
from app.evaluation.deepeval_evaluator import (
    DeepEvalAnswerEvaluator,
    DeepEvalRetrievalEvaluator,
    ...
)
```

When FastAPI started via `uvicorn app.api.main:app`, Python imported `app.api.main`, which imports `app.orchestration.langgraph_graph`, which imports `app.evaluation.evaluate_retrieval`, which is a submodule of `app.evaluation`. Python's package import mechanism executes `app/evaluation/__init__.py` as part of loading the `app.evaluation` package, triggering the top-level `deepeval_evaluator` import. Since `deepeval` is not in the production `requirements.txt`, this raised:

```
ModuleNotFoundError: No module named 'deepeval'
```

This caused the container to fail immediately on startup before serving any requests.

**Fix**

PEP 562 lazy `__getattr__` dispatch in `app/evaluation/__init__.py`. DeepEval symbols are only imported when explicitly accessed by name at runtime. The `TYPE_CHECKING` guard preserves static type-checker visibility. See [D10](#d10--lazy-deepeval-imports) for the full implementation.

**Verification evidence**

Source: `PROJECT_PROGRESS.md` Day 30.

- Docker container started successfully; `GET http://localhost:8000/health` returned `{"status": "healthy"}`.
- `POST /query` with `data/sample_paper.pdf` returned a grounded answer with citations.
- No `ModuleNotFoundError` in Docker logs.

```
pytest tests/test_api.py → 17 passed
pytest tests/test_evaluation_deepeval.py → included in 42 passed, 2 warnings
```

---

### F5 — Groq Rate-Limit Evaluation Failures

**Root cause**

Source: `PROJECT_PROGRESS.md` Day 19, lines 1026–1030; Day 20, lines 1055–1058; Day 25, lines 1193–1198.

DeepEval evaluation uses Groq as the judge LLM (model: `openai/gpt-oss-120b` or `openai/gpt-oss-20b`). Groq enforces a rate limit of approximately 8,000 tokens per minute (TPM) for free-tier or trial access. A DeepEval evaluation run for a single question across multiple metrics (faithfulness, answer relevancy, contextual relevancy, contextual precision, contextual recall) serializes the full retrieved context into multiple judge prompts. For longer retrieved contexts, a single metric call can consume a significant portion of the TPM budget.

Consequences observed:
- Q6 DeepEval metrics: all 5 metrics failed with rate-limit errors → `day25_deepeval_results.json` records these as failures, not scores.
- INSUFFICIENT case: all metrics failed.
- Q1: all 5 metrics succeeded, but 11 rate-limit retries were required across the evaluation run.
- `q1_smoke_test_results.json` cross-configuration run: all 4 configurations returned 0.0 across all metrics — separately caused by the wrong paper being queried (not rate limits), but the 330-second duration reflects rate-limit delays.

**Fix (mitigation, not elimination)**

Source: `PROJECT_PROGRESS.md` Day 20, lines 1047–1052; Day 26 evaluation policy, lines 1278–1292.

1. **Aggregation fix (Day 20)**: `mean_score` is `None` when no valid scores exist. Failed metrics cannot appear as `0.0` scores — they are recorded as failures and displayed as `N/A (N failed)`.
2. **Evaluation policy (Day 26)**: one question at a time; complete all applicable metrics before moving to the next; never intentionally saturate the 30 RPM limit; on 429 stop requests and wait; never convert a rate-limit failure to `null`.
3. **Model switch**: evaluation model changed from `openai/gpt-oss-20b` to `openai/gpt-oss-120b` for improved reliability.

**Verification evidence**

Q1 in `data/evaluation/day25_deepeval_results.json`: faithfulness 1.0, answer relevancy 1.0, contextual precision 1.0, contextual recall 1.0 — all passed. Contextual relevancy 0.3529 — below 0.5 threshold (genuine metric, not a failure).

Q6 and INSUFFICIENT: all metrics recorded as failures due to rate limits. Not reported as valid scores.

**Unresolved limitation**

Rate limits constrain the scale of LLM-as-judge evaluation at the Groq free tier. The Day 25 DeepEval run provides valid metrics for only Q1 (5 metrics) out of 3 intended test cases. A production evaluation environment with higher-capacity Groq or an alternative judge model (e.g., local Ollama) would be required for a comprehensive evaluation sweep.

---

## Summary Table

| ID | Decision / Failure | Type | Key Evidence |
|---|---|---|---|
| D1 | Dense vs. hybrid retrieval | Decision | Day 17: dense avg RR 0.78125 → hybrid RRF 0.85417 (+9.33%) |
| D2 | BM25 + RRF design | Decision | `bm25_index.py` class constants; Day 15 implementation log |
| D3 | Cross-encoder reranking | Decision | Day 17: hybrid RRF avg RR 0.7083 → reranked 0.8750 (+23.53%) |
| D4 | CLIP visual + multimodal reranker | Decision | Day 21: Q6 image enters top-5 with MultimodalReranker |
| D5 | Relationship expansion | Decision | Day 18: Q6 caption→image expansion; one-hop only |
| D6 | LangGraph orchestration | Decision | Day 24: typed GraphState, conditional error routing |
| D7 | Candidate pool optimization | Decision | Day 28: -334 ms (-42.5%), MRR preserved at 0.9375 |
| D8 | In-memory FAISS | Decision | Day 5, 28: FAISS retrieval 0.35 ms; rebuild per request by design |
| D9 | Request-local candidate count | Decision | Day 29: removed `self._last_candidate_count` |
| D10 | Lazy DeepEval imports | Decision | Day 30: PEP 562 `__getattr__`; API startup unblocked |
| D11 | Structured logging / deferred LangSmith | Decision | Day 29: 3 named loggers, JSON events, no external service |
| F1 | CLIP normalization bug | Failure | Day 22–23: Q6 image dropped rank 5→6; fixed with explicit neutral 0.5 |
| F2 | FastAPI exception-handler TypeError | Failure | Day 27: incorrect handler signature; corrected per FastAPI protocol |
| F3 | Shared mutable candidate-count race | Failure | Day 29: removed instance attribute; returned via tuple |
| F4 | Eager DeepEval import / startup failure | Failure | Day 30: `ModuleNotFoundError`; fixed with lazy `__getattr__` |
| F5 | Groq rate-limit evaluation failures | Failure | Day 19–25: Q6/INSUFFICIENT metrics failed; aggregation corrected |

