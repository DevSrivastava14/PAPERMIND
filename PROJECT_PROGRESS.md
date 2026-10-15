# Project Progress & Tracking

## Project Goal
Build a multimodal scientific research paper assistant that answers questions about research PDFs using grounded evidence and citations, with eventual support for text, figures, and tables.

---

## 33-Day Planned Roadmap

- **Days 1–7: Fundamentals and Text RAG**
  - Architecture, project structure, PDF parsing, text extraction, chunking, embeddings, vector indexing, retrieval pipeline, and baseline question-answering with citations.
- **Days 8–14: Multimodal Extraction**
  - Extracting images, figures, and tables; visual encoding/multimodal embeddings; table parsing; multimodal document representation.
- **Days 15–17: Hybrid Retrieval and Reranking**
  - Sparse (BM25/keyword) + dense retrieval, reciprocal rank fusion (RRF), cross-encoder reranking.
- **Days 18–22: Evaluation and Ragas**
  - Ground truth dataset construction, retrieval and generation metrics (faithfulness, answer relevance, context precision/recall) with Ragas, automated evaluation harness.
- **Days 23–28: Security, Guardrails, Langfuse, and Performance**
  - Prompt injection defense, guardrails, observability and tracing with Langfuse, caching, latency and throughput optimization.
- **Days 29–33: Testing, Docker, Documentation, and Interview Preparation**
  - End-to-end integration tests, Docker containerization, comprehensive documentation, portfolio and interview preparation.

---

## Planned Technology Stack

| Component | Technology | Status |
| :--- | :--- | :--- |
| **Language / Runtime** | Python 3.13.5 | Verified (Installed in `.venv`) |
| **PDF Parsing** | PyMuPDF (`pymupdf==1.28.2`) | Verified (Installed in `.venv`) |
| **Testing** | pytest (`pytest==9.1.1`) | Verified (Installed in `.venv`) |
| **API Framework** | FastAPI | Planned (Unverified) |
| **Data Validation** | Pydantic | Planned (Unverified) |
| **Embeddings** | Sentence Transformers (`sentence-transformers==6.1.0`) | Verified (Installed in `.venv`) |
| **Vector Storage** | FAISS (`faiss-cpu==1.15.1`) | Verified (Installed in `.venv`) |
| **LLM Inference** | LLM Provider or Local Alternative | Planned (Unverified) |
| **Evaluation** | Ragas | Planned (Unverified) |
| **Observability** | Langfuse | Planned (Unverified) |

---

## Current State

- **Status:** Day 11 Complete
- **Date:** October 15, 2026
- **Environment:** Windows, Python 3.13.5, active `.venv`
- **Version Control:** Git initialized locally; no commits made (files uncommitted/staged as requested)
- **Configuration & Dependencies:**
  - `requirements.txt`: Pinned to `pymupdf==1.28.2`, `pytest==9.1.1`, `sentence-transformers==6.1.0`, and `faiss-cpu==1.15.1`
  - `.gitignore`: Configured and verified (ignores `.venv/`, pytest/Python caches, `data/*.pdf`, `data/processed/`, `data/uploads/`, `.env*` except `.env.example`, and logs)
  - `.env.example`: Contains `APP_ENV=development`
  - `README.md`: Intentionally kept empty until project completion (ComputeGrid workflow)

---

## Daily Log

### Day 1 — October 10, 2026 (Completed)
- **Completed Tasks:**
  - Initialized project tracking document (`PROJECT_PROGRESS.md`) as the single source of truth.
  - Initialized local Git repository.
  - Verified local environment with Python 3.13.5 and `.venv`.
  - Pinned `requirements.txt` to `pymupdf==1.28.2` and `pytest==9.1.1`.
  - Added and verified `.gitignore` for environment, cache, local data, and log files.
  - Created `.env.example` with `APP_ENV=development`.
  - Implemented initial PDF parser in `app/ingestion/pdf_parser.py` with `extract_pdf_pages(pdf_path)` returning 1-based `page_number`, `text`, and boolean `has_text` (true only if non-whitespace characters exist). Includes error handling for missing files (`FileNotFoundError`), invalid/unreadable PDFs (`ValueError`), and ensures safe document closure.
  - Created test suite in `tests/test_pdf_parser.py` covering multi-page extraction, 1-based page numbering, empty/whitespace pages, missing files, and invalid PDFs with programmatically generated PDFs.
  - Executed tests: `python -m pytest -q` passed (5 passed).
  - Validated parser against `data/sample_paper.pdf`: 7 pages parsed, all 7 contain extractable text. Observed running headers/layout artifacts and line-wrapped references; accepted as baseline extraction limitations without adding ad-hoc cleanup.
- **Changed Files:**
  - `app/ingestion/pdf_parser.py` (Created)
  - `tests/test_pdf_parser.py` (Created)
  - `requirements.txt` (Updated with pinned versions)
  - `.gitignore` (Updated)
  - `.env.example` (Verified)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `python -m pytest -q`: 5 passed in 0.25s
  - `git status --short`: untracked files reported, 0 commits made
- **Decisions Made:**
  - Preserved raw PyMuPDF extraction output without heuristic cleaning to keep baseline parser modular and predictable.
  - Kept `README.md` empty until project completion per ComputeGrid workflow.
- **Blockers:**
  - None.

### Day 2 — October 11, 2026 (Completed)
- **Completed Tasks:**
  - Created lightweight standard-library dataclasses in `app/ingestion/models.py`: `PageContent` (`page_number`, `text`, `has_text`) and `ResearchPaperDocument` (`filename`, `total_pages`, `pages`).
  - Added public ingestion interface `parse_research_paper(pdf_path)` in `app/ingestion/pdf_parser.py` reusing existing `extract_pdf_pages()` to avoid logic duplication.
  - Preserved source filename (via `Path(pdf_path).name`), total page count calculation, and wrapped page dictionaries into `PageContent` objects.
  - Maintained 100% backward compatibility for `extract_pdf_pages()` and preserved standard error handling (`FileNotFoundError`, `ValueError`).
  - Added 4 new tests in `tests/test_pdf_parser.py` covering document structure, metadata, blank page preservation, error raising, and a fixture-safe smoke test against `data/sample_paper.pdf`.
  - Executed test suite in `.venv`: `.\.venv\Scripts\python -m pytest -q` passed (9 passed).
- **Changed Files:**
  - `app/ingestion/models.py` (Created)
  - `app/ingestion/pdf_parser.py` (Modified)
  - `tests/test_pdf_parser.py` (Modified)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\.venv\Scripts\python -m pytest -q`: 9 passed in 0.52s
  - `git status --short`: `M app/ingestion/pdf_parser.py`, `M tests/test_pdf_parser.py`, `?? app/ingestion/models.py`
- **Decisions Made:**
  - Implemented standard-library dataclasses to avoid unnecessary external dependencies or framework overhead while ensuring type safety and clean interfaces.
  - Kept raw text extraction unchanged to preserve baseline parser modularity.
  - Isolated file basename in `filename` attribute for cleaner downstream citation references.
- **Remaining Limitations:**
  - PDF layout artifacts (headers, footers, hyphenation) remain in raw page text.
  - Extraction remains text-only; figures and tables will be handled in multimodal milestone (Days 8–14).
- **Blockers:**
  - None.

### Day 3 — October 11, 2026 (Completed)
- **Completed Tasks:**
  - Created `app/chunking/models.py` defining `TextChunk` dataclass (`chunk_id`, `text`, `source_filename`, `chunk_index`, `page_numbers`).
  - Implemented `TextChunker` class and convenience function `chunk_document()` in `app/chunking/text_chunker.py` using a deterministic sliding window (`chunk_size=1000`, `chunk_overlap=150`).
  - Added strict parameter validation enforcing `chunk_size > 0`, `chunk_overlap >= 0`, and `chunk_overlap < chunk_size` with informative `ValueError` exceptions.
  - Implemented accurate 1-based page provenance tracking using character offset spans; verified page separators (`\n\n`) do not cause false page attribution.
  - Handled edge cases: skipped blank/whitespace pages, returned empty lists for unextractable documents, and preserved source text and final short chunks without truncation.
  - Generated deterministic chunk IDs (`f"{filename}#chunk_{index}"`) with sequential 0-based indices.
  - Re-exported public chunking interface in `app/chunking/__init__.py`.
  - Created test suite in `tests/test_text_chunker.py` covering basic sliding window with overlap, exact chunk boundaries, final short chunk integrity, short documents, blank page skipping, multi-page boundary spanning, deterministic IDs, parameter validation errors, and smoke test on `data/sample_paper.pdf`.
  - Executed tests in `.venv`: `.\.venv\Scripts\python -m pytest -q` passed (17 passed).
- **Changed Files:**
  - `app/chunking/__init__.py` (Created)
  - `app/chunking/models.py` (Created)
  - `app/chunking/text_chunker.py` (Created)
  - `tests/test_text_chunker.py` (Created)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\.venv\Scripts\python -m pytest -q`: 17 passed in 0.56s
  - `git status --short`: `M PROJECT_PROGRESS.md`, `?? app/chunking/`, `?? tests/test_text_chunker.py`
- **Decisions Made:**
  - Created dedicated `app/chunking/` module to isolate text processing from PDF ingestion.
  - Maintained zero external dependencies by using standard-library dataclasses and pure Python logic.
  - Mapped page provenance via explicit character offsets rather than string matching to guarantee 100% accuracy and prevent false attribution across page separators.
- **Remaining Limitations:**
  - Pure character-based chunking may split words or sentences mid-boundary; semantic/sentence-aware boundary splitting is deferred.
  - Multimodal chunks (tables, figures) deferred to Days 8–14.
- **Blockers:**
  - None.

### Day 4 — October 12, 2026 (Completed)
- **Completed Tasks:**
  - Pinned `sentence-transformers==6.1.0` in `requirements.txt` for Python 3.13.5 Windows compatibility, preserving `pymupdf==1.28.2` and `pytest==9.1.1`.
  - Created dedicated `app/embeddings/` module with clean interface and model structures.
  - Created `app/embeddings/models.py` defining `EmbeddedChunk` and `EmbeddingResult` preserving 2D float32 vectors, chunk IDs, separate metadata dictionaries, and original `TextChunk` instances.
  - Created `app/embeddings/base.py` defining abstract `BaseEmbedder` interface with `dimension`, `model_name`, `embed_texts()`, and default metadata-preserving `embed_chunks()`.
  - Implemented `SentenceTransformerEmbedder` in `app/embeddings/sentence_transformer_embedder.py` using `sentence-transformers/all-MiniLM-L6-v2` (dimension 384):
    - Sequential batching via configurable `batch_size`.
    - Strict vector validation: 2D array, count check, dimension check, and finite numerical values check (`np.isfinite()`).
    - Handled empty input without crashing, returning empty `(0, dimension)` float32 arrays.
    - Model caching via `get_embedder()` and `reset_embedder()` to prevent repeated model loading across calls.
    - Testable dependency injection support accepting pre-instantiated or mocked models via `model` parameter.
  - Re-exported public embedding interface in `app/embeddings/__init__.py`.
  - Created test suite in `tests/test_embedder.py` (10 tests) covering empty input, output dimensions, vector count, chunk-to-vector mapping, batching call counts, non-finite numerical outputs (NaN and Inf), dimension mismatch, vector count mismatch, invalid batch sizes, and model caching reuse. All tests use mocked models without network access or model downloads.
  - Executed full test suite: 27 passed in 12.59s.
- **Changed Files:**
  - `requirements.txt` (Modified: added `sentence-transformers==6.1.0`)
  - `app/embeddings/__init__.py` (Created)
  - `app/embeddings/base.py` (Created)
  - `app/embeddings/models.py` (Created)
  - `app/embeddings/sentence_transformer_embedder.py` (Created)
  - `tests/test_embedder.py` (Created)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\.venv\Scripts\python -m pytest -v`: 27 passed in 12.59s (10 in `test_embedder.py`, 9 in `test_pdf_parser.py`, 8 in `test_text_chunker.py`)
  - `git status --short`: `M PROJECT_PROGRESS.md`, `M requirements.txt`, `?? app/embeddings/`, `?? tests/test_embedder.py`
- **Decisions Made:**
  - Pinned `sentence-transformers==6.1.0` which has native wheels for Python 3.13 on Windows.
  - Separated vector array `vectors` (2D float32) from `chunk_ids` and `metadata` in `EmbeddingResult` to align directly with FAISS index requirements (`index.add()`) while keeping 1:1 provenance.
  - Enforced strict numerical validation (`np.isfinite()`) to catch upstream model errors early before passing vectors to C++ FAISS bindings.
  - Provided module-level `get_embedder()` caching and constructor injection (`model=...`) to ensure unit tests run offline in milliseconds and production code avoids repeated weights loading.
- **Remaining Limitations:**
  - Initial `torch` DLL load in Windows Python 3.13 incurs a one-time startup latency (~10–12s on fresh process spawn).
  - Downstream vector storage (FAISS indexing) and similarity search deferred to Day 5.
- **Blockers:**
  - None.

### Day 5 — October 12, 2026 (Completed)
- **Completed Tasks:**
  - Pinned `faiss-cpu==1.15.1` in `requirements.txt` for Python 3.13.5 Windows compatibility, preserving all existing dependencies.
  - Created dedicated `app/indexing/` module with public interface for FAISS vector indexing and similarity search.
  - Created `app/indexing/models.py` defining `SearchResult` dataclass (`chunk`, `score`, `chunk_id`, `source_filename`, `page_numbers`).
  - Implemented `FAISSIndex` class in `app/indexing/faiss_index.py`:
    - Utilizes FAISS `IndexFlatIP` with L2-normalized vectors so search inner product mathematically matches cosine similarity.
    - Validates vector dimensions, vector counts, finite numerical values (`np.isfinite()`), and input consistency before adding to FAISS.
    - Preserves bidirectional mappings: internal integer ID -> `TextChunk` and `chunk_id` -> internal ID.
    - Implemented `search_by_vector(query_vector, top_k)` with query L2-normalization and filtering of negative FAISS placeholder indices (`-1`).
    - Implemented `search(query, top_k, embedder)` embedding text queries using the existing `BaseEmbedder` interface.
    - Handled edge cases: empty indexes, empty or whitespace queries, non-positive `top_k`, and fewer than `top_k` indexed items without crashing.
    - Provided factory method `FAISSIndex.from_embedding_result()` and convenience function `build_index()`.
  - Re-exported public indexing interface in `app/indexing/__init__.py`.
  - Created test suite in `tests/test_indexing.py` (9 tests) covering empty index search, empty text query search, invalid `top_k`, adding empty embedding results, exact cosine ranking and score verification, provenance and ID-to-chunk lookups, fewer-than-k result filtering, text search with mock embedders, and invalid input validation (NaN, Inf, dimension mismatch).
  - Executed full test suite in `.venv`: 36 passed in 10.56s.
- **Changed Files:**
  - `requirements.txt` (Modified: added `faiss-cpu==1.15.1`)
  - `app/indexing/__init__.py` (Created)
  - `app/indexing/models.py` (Created)
  - `app/indexing/faiss_index.py` (Created)
  - `tests/test_indexing.py` (Created)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\.venv\Scripts\python -m pytest -v`: 36 passed in 10.56s (9 in `test_indexing.py`, 10 in `test_embedder.py`, 9 in `test_pdf_parser.py`, 8 in `test_text_chunker.py`)
  - `git status --short`: `M PROJECT_PROGRESS.md`, `M requirements.txt`, `?? app/indexing/`, `?? tests/test_indexing.py`
- **Decisions Made:**
  - Used FAISS `IndexFlatIP` coupled with L2-normalization to achieve exact cosine similarity ranking while avoiding Euclidean distance score inversions.
  - Implemented safe zero-norm handling (`np.where(norm == 0, 1.0, norm)`) during normalization to avoid division by zero.
  - Filtered out negative indices (`idx < 0`) in search results to cleanly handle cases where `top_k` exceeds the total number of items in the index without leaking FAISS sentinel values.
  - Preserved the existing `TextChunk` and `EmbeddingResult` data models without unnecessary duplication.
- **Remaining Limitations:**
  - In-memory index only; disk persistence (`faiss.write_index`) is deferred.
  - Dense semantic retrieval only; sparse keyword search (BM25) and hybrid retrieval deferred to Days 15–17.
- **Blockers:**
  - None.


### Day 6 — October 13, 2026 (Completed)
- **Completed Tasks:**
  - **Retrieval models:** Added `app/retrieval/models.py` defining `Citation` (`citation_label`, `source_filename`, `page_numbers`, `chunk_id`, `score`) and `RetrievedContext` (`formatted_context`, `citations`, `search_results`) dataclasses.
  - **Citation context assembly:** Added `app/retrieval/assembler.py` with `CitationContextAssembler`, `format_page_reference()`, and convenience function `assemble_context()`. Formats retrieved chunks into clearly separated evidence blocks with citation markers (`[1] Source: filename (Page X)\n<chunk text>`).
  - **Retrieval pipeline:** Added `app/retrieval/pipeline.py` with `RetrievalPipeline` class and convenience function `build_pipeline()`. Reuses existing `FAISSIndex.search()` interface with optional injected embedder, default `top_k`, and per-call override.
  - **End-to-end workflow:** Validates query string, retrieves relevant chunks via existing FAISS index, and seamlessly formats search results into prompt-ready evidence blocks.
  - **Citation provenance:** Preserves original source filename, exact 1-based PDF page numbers (never renumbered or invented), chunk ID, similarity score, and sequential citation labels (`[1]`, `[2]`, ...).
  - **Edge cases:** Handled and verified empty search results (returns empty context without crashing), empty/whitespace queries (returns empty results without querying index), and invalid non-positive `top_k` values (raises descriptive `ValueError`).
  - **Module exports:** Created `app/retrieval/__init__.py` re-exporting all public retrieval models and functions.
  - **Tests:** Added comprehensive test suite in `tests/test_retrieval.py` (8 tests) using synthetic `SearchResult` objects and mock dependencies covering page reference formatting, citation/page metadata, normal evidence assembly, empty results, invalid `top_k`, whitespace queries, and query-to-context flow.
  - **Test Suite Status:** Full test suite passes completely with **44 passed** in 10.19s (8 in `test_retrieval.py`, 9 in `test_indexing.py`, 10 in `test_embedder.py`, 9 in `test_pdf_parser.py`, 8 in `test_text_chunker.py`).
- **Changed Files:**
  - `app/retrieval/__init__.py` (Created)
  - `app/retrieval/models.py` (Created)
  - `app/retrieval/assembler.py` (Created)
  - `app/retrieval/pipeline.py` (Created)
  - `tests/test_retrieval.py` (Created)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\.venv\Scripts\python -m pytest -q`: 44 passed in 10.19s
  - `git status --short`: `?? app/retrieval/`, `?? tests/test_retrieval.py`, `M PROJECT_PROGRESS.md`
- **Decisions Made:**
  - Preserved original 1-based page numbers from chunk provenance; never fabricated or altered source page numbers.
  - Reused existing `SearchResult`, `FAISSIndex`, and `BaseEmbedder` abstractions directly without modifying chunking, embedding, or indexing code.
  - Used deterministic 1-based sequential citation markers matching structured citations for clean prompt injection and verifiable attribution.
- **Remaining Limitations:**
  - Dense semantic retrieval only; sparse BM25 and hybrid reranking deferred to Days 15–17.
  - In-memory index without document-level metadata filtering; full metadata filtering deferred to upcoming milestones.
- **Blockers:**
  - None.


### Day 7 — October 13, 2026 (Completed)
- **Completed Tasks:**
  - **Generation models:** Added `app/generation/models.py` defining `AnswerResult` dataclass (`answer`, `citations`, `query`, `has_context`, `model_name`).
  - **Provider-independent interface:** Added `app/generation/base.py` with abstract `BaseGenerator` class (`model_name` property, `generate(prompt)` method). Any LLM provider can be plugged in by subclassing `BaseGenerator` without modifying retrieval or orchestration logic.
  - **Grounded prompt builder:** Added `app/generation/prompt_builder.py` with `build_grounded_prompt(query, context)`. Embeds a system instruction requiring the model to answer using only supplied evidence with inline citation markers, explicitly forbids fabrication, and delimits the evidence block (`--- EVIDENCE ---` / `--- END OF EVIDENCE ---`) from the question to reduce prompt-injection surface.
  - **Answer generation orchestration:** Added `app/generation/answer_generator.py` with `AnswerGenerator` and convenience factory `build_answer_generator()`. Guards empty context by returning a clear insufficient-evidence response without calling the generator. On non-empty context, builds the prompt, invokes the generator, and returns `AnswerResult` with the answer and preserved `Citation` list copied from `RetrievedContext`.
  - **Module exports:** Created `app/generation/__init__.py` re-exporting all public generation symbols.
  - **Tests:** Added `tests/test_generation.py` (19 tests) using `FakeGenerator` — a deterministic in-process stub with no external LLM calls, no network access, and no model downloads. Covers: system instruction presence, evidence delimiter placement, formatted context embedding, question forwarding, prompt section ordering, `Answer:` marker, evidence-inside-delimiters check, `BaseGenerator` abstract-class guard, invalid generator type at construction, factory function, successful answer result structure, prompt forwarded to generator, citation metadata integrity, citation list independence from context, empty-context insufficient-evidence return, empty-context generator not called, empty/whitespace query `ValueError`, whitespace query generator not called, generator exception propagation.
  - **Full test suite:** **63 passed** in 11.42s (19 in `test_generation.py`, 8 in `test_retrieval.py`, 9 in `test_indexing.py`, 10 in `test_embedder.py`, 9 in `test_pdf_parser.py`, 8 in `test_text_chunker.py`).
- **Changed Files:**
  - `app/generation/__init__.py` (Created)
  - `app/generation/models.py` (Created)
  - `app/generation/base.py` (Created)
  - `app/generation/prompt_builder.py` (Created)
  - `app/generation/answer_generator.py` (Created)
  - `tests/test_generation.py` (Created)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\.venv\Scripts\python -m pytest -q`: 63 passed in 11.42s
  - `git status --short`: `?? app/generation/`, `?? tests/test_generation.py`, `M PROJECT_PROGRESS.md`
- **Decisions Made:**
  - Defined `BaseGenerator` as an abstract class rather than a `Protocol` so implementors receive a clear `TypeError` at construction if the interface is not satisfied — consistent with Days 4–6 patterns (`BaseEmbedder`).
  - Delimited the evidence block with `--- EVIDENCE ---` / `--- END OF EVIDENCE ---` lines to reduce prompt-injection risk: instructions embedded inside retrieved document text cannot escape into the system instruction region.
  - Returned `list(context.citations)` (a shallow copy) from `AnswerResult` so the caller cannot mutate result citations and inadvertently affect the originating context.
  - The empty-context guard returns immediately before calling `build_grounded_prompt`, ensuring no partially constructed prompt is ever passed to the generator when no evidence exists.
- **Remaining Limitations:**
  - No real LLM provider implemented; a concrete `BaseGenerator` subclass must be written before live inference is possible.
  - Generator is called once per query; streaming responses and multi-turn conversation are deferred.
  - No per-document metadata filtering in retrieval; deferred to upcoming milestones.
- **Blockers:**
  - None.


### Day 8 — October 14, 2026 (Completed)
- **Completed Tasks:**
  - **Multimodal image extraction:** Implemented embedded image/figure extraction from research PDFs using PyMuPDF (`pymupdf`).
  - **New module:** Created `app/multimodal/` module with:
    - `app/multimodal/models.py`: Defines `ExtractedImage` dataclass (with aliases `VisualElement` and `ExtractedVisualElement`) representing an extracted visual element with `image_id`, `source_filename`, 1-based `page_number`, 0-based `image_index`, `image_data` (bytes), `image_path` (reference), format, and dimensions.
    - `app/multimodal/image_extractor.py`: Defines `ImageExtractor` and clean `extract_images(pdf_path, output_dir=None)` function detecting and extracting embedded images across pages.
    - `app/multimodal/__init__.py`: Re-exports public multimodal classes and functions.
  - **Page provenance & deterministic indexing:** Preserved exact 1-based PDF page numbers (`page_number >= 1`) and deterministic image indexing and identity (`f"{path.name}#page_{page_idx}_img_{img_idx}"`).
  - **Edge cases handled:** Handled PDFs with no images cleanly returning an empty list `[]`, missing file raising `FileNotFoundError`, and invalid/corrupted PDFs raising `ValueError`.
  - **Sample paper extraction:** Extracted 13 embedded images/figures from `data/sample_paper.pdf`.
  - **Manual verification:** Confirmed extracted images match the corresponding figures/pages in `sample_paper.pdf`.
  - **Integrity preserved:** Existing ingestion, chunking, embeddings, indexing, retrieval, and generation code was not modified.
  - **Tests:** Added 12 focused unit and integration tests in `tests/test_multimodal.py`.
  - **Full test suite:** **75 passed** in 11.79s (12 in `test_multimodal.py`, 19 in `test_generation.py`, 8 in `test_retrieval.py`, 9 in `test_indexing.py`, 10 in `test_embedder.py`, 9 in `test_pdf_parser.py`, 8 in `test_text_chunker.py`).
- **Changed Files:**
  - `app/multimodal/__init__.py` (Created)
  - `app/multimodal/models.py` (Created)
  - `app/multimodal/image_extractor.py` (Created)
  - `tests/test_multimodal.py` (Created)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\.venv\Scripts\python -m pytest -q`: 75 passed in 11.79s
- **Decisions Made:**
  - Reused existing PyMuPDF (`pymupdf`) dependency for image extraction without introducing unnecessary new frameworks.
  - Implemented `ExtractedImage` with strict validation for 1-based `page_number`, non-negative `image_index`, non-empty IDs, and `bytes` payload.
  - Maintained backward compatibility via aliases `VisualElement` and `ExtractedVisualElement` and accessor properties `image_bytes`, `path`, and `size`.
  - Kept image extraction in-memory by default, with optional file persistence via `output_dir`.
- **Remaining Limitations:**
  - Table extraction, visual encoding/multimodal embeddings, and OCR deferred to upcoming days in the multimodal milestone (Days 8–14).
- **Blockers:**
  - None.


### Day 9 — October 14, 2026 (Completed)
- **Completed Tasks:**
  - **Equation/formula model:** Created `ExtractedEquation` dataclass in `app/multimodal/models.py` (with aliases `EquationElement` and `ExtractedFormula`) representing mathematical expressions with `equation_id`, `source_filename`, 1-based `page_number`, 0-based `equation_index`, `text`, `equation_label`, `bbox`, and `is_display`.
  - **Equation extraction component:** Implemented `app/multimodal/equation_extractor.py` containing `EquationExtractor` and clean `extract_equations(pdf_path)` function using PyMuPDF (`pymupdf`).
  - **Pattern classification & heuristics:** Built robust heuristic classifier distinguishing mathematical formulas from prose:
    - Numbered equations ending with labels like `(1)`, `(2.1)`, `[1]`, `Eq. (1)`.
    - Mathematical relations and operators (`=`, `≈`, `≤`, `≥`, `≠`, `:=`, `∝`, etc.).
    - Unicode math symbols (`∑`, `∫`, `∏`, `√`, `∂`, `∇`, Greek letters) and LaTeX macros (`\sum`, `\int`, `\frac`, etc.).
    - Multiline equation continuation detection with bounding box unioning.
    - False positive rejection for prose sentences and figure/table captions (`Figure 1:`, `Table 1:`).
  - **Module exports:** Updated `app/multimodal/__init__.py` to re-export all equation extraction models and functions.
  - **Provenance & deterministic identity:** Preserves exact 1-based PDF page provenance, deterministic sequential indexing (`equation_index`), and unique IDs (`{filename}#page_{page}_eq_{index}`).
  - **Edge cases handled:** Handled PDFs with no equations cleanly returning an empty list `[]`, missing file raising `FileNotFoundError`, and invalid/corrupted PDFs raising `ValueError`.
  - **Preserved existing pipeline:** Existing ingestion, chunking, embeddings, FAISS, retrieval, and generation code was not modified.
  - **Tests:** Added 11 focused unit and integration tests in `tests/test_equation_extractor.py`.
  - **Full test suite:** **86 passed** in 16.36s (11 in `test_equation_extractor.py`, 12 in `test_multimodal.py`, 19 in `test_generation.py`, 8 in `test_retrieval.py`, 9 in `test_indexing.py`, 10 in `test_embedder.py`, 9 in `test_pdf_parser.py`, 8 in `test_text_chunker.py`).
  - **Manual verification against `data/sample_paper.pdf`:** Confirmed 0 equations detected across all 7 pages without error or false positives, accurately reflecting the survey nature of the sample paper.
- **Changed Files:**
  - `app/multimodal/models.py` (Modified)
  - `app/multimodal/equation_extractor.py` (Created)
  - `app/multimodal/__init__.py` (Modified)
  - `tests/test_equation_extractor.py` (Created)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\.venv\Scripts\python -m pytest tests/test_equation_extractor.py -v`: 11 passed in 0.70s
  - `.\.venv\Scripts\python -m pytest -q`: 86 passed in 16.36s
- **Decisions Made:**
  - Reused PyMuPDF (`pymupdf`) text block/span analysis without introducing heavy math/OCR dependencies (like pix2tex or sympy).
  - Implemented `ExtractedEquation` with strict validation for 1-based `page_number`, non-negative `equation_index`, non-empty IDs, and non-empty equation text.
  - Provided property accessors (`equation_text`, `expression`, `content`) and aliases (`EquationElement`, `ExtractedFormula`) for flexible downstream consumption.
  - Handled multiline equations by checking for leading operators and unioning spatial bounding boxes.
- **Remaining Limitations:**
  - Extraction relies on text-layer elements; purely rasterized or bitmap-embedded equations require visual/OCR extraction (deferred to future multimodal days).
  - Table parsing, caption association, and unified multimodal representation deferred to Days 10–14.
- **Blockers:**
  - None.


### Day 10 — October 15, 2026 (Completed)
- **Completed Tasks:**
  - **Structured table model:** Created `ExtractedTable` dataclass in `app/multimodal/models.py` (with aliases `TableElement` and `ExtractedTableElement`) representing structured tabular data with `table_id`, `source_filename`, 1-based `page_number`, 0-based `table_index`, `rows` (2D list of cells), `num_rows`, `num_cols`, `content` (markdown/text representation), `bbox` (spatial bounding box in PDF points), and `headers`.
  - **Table extraction component:** Implemented `app/multimodal/table_extractor.py` containing `TableExtractor` and clean `extract_tables(pdf_path, **kwargs)` function using existing PyMuPDF (`pymupdf`) table detection capabilities without adding heavy external OCR or table extraction frameworks.
  - **Module exports:** Updated `app/multimodal/__init__.py` to re-export all table extraction models and functions (`ExtractedTable`, `TableElement`, `ExtractedTableElement`, `TableExtractor`, `extract_tables`).
  - **Page provenance & deterministic identity:** Preserves exact 1-based PDF page provenance (`page_number >= 1`), deterministic sequential table indexing (`table_index`), and unique IDs (`{source_filename}#page_{page_number}_table_{table_index}`).
  - **Edge cases handled:** Handled PDFs with no tables cleanly returning an empty list `[]`, missing file raising `FileNotFoundError`, and invalid/corrupted PDFs raising `ValueError`.
  - **Preserved existing pipeline:** Existing ingestion, chunking, embeddings, FAISS, retrieval, generation, and equation/image extraction code was not modified.
  - **Tests:** Added 13 focused unit and integration tests in `tests/test_table_extractor.py`.
  - **Test results:** 13 passed in `test_table_extractor.py`, 23 passed in `test_multimodal.py` + `test_equation_extractor.py`.
  - **Verification against `data/sample_paper.pdf`:** Confirmed 0 tables detected across all 7 pages without error, accurately reflecting the absence of tabular grids in the sample paper.
- **Changed Files:**
  - `app/multimodal/models.py` (Modified)
  - `app/multimodal/table_extractor.py` (Created)
  - `app/multimodal/__init__.py` (Modified)
  - `tests/test_table_extractor.py` (Created)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\.venv\Scripts\python -m pytest tests/test_table_extractor.py -v`: 13 passed in 2.18s
  - `.\.venv\Scripts\python -m pytest tests/test_multimodal.py tests/test_equation_extractor.py -v`: 23 passed in 0.97s
- **Decisions Made:**
  - Reused PyMuPDF's native `find_tables()` API (`TableFinder` and `Table` objects) without introducing heavy table frameworks (e.g., camelot, tabula, paddleocr).
  - Implemented `ExtractedTable` with strict validation for 1-based `page_number`, non-negative `table_index`, non-empty IDs, and list-of-lists `rows`.
  - Built-in automatic markdown formatting (`_rows_to_markdown`) when custom markdown content is not supplied.
  - Provided property accessors (`table_content`, `text`, `markdown`, `row_count`, `col_count`, `cells`, `data`, `columns`) and aliases (`TableElement`, `ExtractedTableElement`) for flexible downstream consumption.
- **Remaining Limitations:**
  - Captions, table-to-caption relationships, visual embeddings, and unified multimodal representation deferred to Days 11–14.
- **Blockers:**
  - None.


### Day 11 — October 15, 2026 (Completed)
- **Completed Tasks:**
  - **Caption model:** Added `ExtractedCaption` dataclass in `app/multimodal/models.py` (with aliases `CaptionElement` and `ExtractedCaptionElement`) representing a figure or table caption with `caption_id`, `source_filename`, 1-based `page_number`, 0-based `caption_index`, `caption_type` (`'figure'` or `'table'`), `text`, `label`, and `bbox`. Added module-level sentinel constants `CAPTION_TYPE_FIGURE = 'figure'` and `CAPTION_TYPE_TABLE = 'table'`.
  - **Relationship model:** Added `CaptionMatch` dataclass representing a lightweight, deterministic spatial link between a visual element (`image_id` / `table_id`) and its caption (`caption_id`), carrying `proximity_pts`, `source_filename`, and `page_number`.
  - **Caption extraction component:** Implemented `app/multimodal/caption_extractor.py` with `CaptionExtractor`, `CaptionExtractionResult`, and convenience function `extract_captions(pdf_path, images=None, tables=None)` using PyMuPDF text-block scanning and spatial bbox proximity.
  - **Detection strategy:** Two-pass per page — (1) scan text blocks whose first line matches a `Figure|Fig.|Table|Tab.` + number pattern; (2) spatial association pairs each caption with the nearest image block (for figure captions) or supplied `ExtractedTable` bbox (for table captions) within a configurable `max_proximity_pts` threshold (default 72 pts).
  - **Prefix support:** Detects `Figure N`, `Fig. N`, `FIG N`, `Table N`, `Tab. N`, `TAB N` — normalised to `'figure'`/`'table'` type and canonical `'Figure N'`/`'Table N'` label.
  - **Module exports:** Updated `app/multimodal/__init__.py` to re-export all caption symbols (`CAPTION_TYPE_FIGURE`, `CAPTION_TYPE_TABLE`, `ExtractedCaption`, `CaptionElement`, `ExtractedCaptionElement`, `CaptionMatch`, `CaptionExtractionResult`, `CaptionExtractor`, `extract_captions`).
  - **Sample paper verification:** `data/sample_paper.pdf` → 3 figure captions detected (FIGURE 1, FIGURE 2, FIGURE 3) across pages 2 and 5, all correctly classified as `'figure'`.
  - **Tests:** Added 37 focused tests in `tests/test_caption_extractor.py`.
  - **Full multimodal suite:** 73 passed in 3.35s (37 new + 36 existing Days 8-10 tests).
- **Changed Files:**
  - `app/multimodal/models.py` (Modified)
  - `app/multimodal/caption_extractor.py` (Created)
  - `app/multimodal/__init__.py` (Modified)
  - `tests/test_caption_extractor.py` (Created)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\.venv\Scripts\python -m pytest tests/test_caption_extractor.py -v`: **37 passed in 0.98s**
  - `.\.venv\Scripts\python -m pytest tests/test_caption_extractor.py tests/test_table_extractor.py tests/test_equation_extractor.py tests/test_multimodal.py -v`: **73 passed in 3.35s**
- **Decisions Made:**
  - Used PyMuPDF `page.get_text('dict')` (already used by `EquationExtractor`) — no new dependencies.
  - Figure captions matched against PDF layout image blocks (type=1) rather than `ExtractedImage` (which carries no bbox), keeping the extractor self-contained when no images are supplied.
  - Table caption matching requires caller-supplied `ExtractedTable` objects with populated `bbox`, enabling optional linkage without hard-coupling the extractors.
  - `CaptionMatch.proximity_pts` is the raw vertical gap in PDF points — deterministic, reproducible, and threshold-configurable downstream.
  - `max_proximity_pts` defaults to 72 pts (1 inch) to handle typical typographic spacing while preventing cross-column misattribution.
- **Remaining Limitations:**
  - Multi-line captions spanning across two PDF text blocks are not merged.
  - Visual encoding, multimodal embeddings, and unified multimodal representation deferred to Days 12-14.
- **Blockers:**
  - None.

---

## Handoff

- **Status:** Day 11 is complete.
- **Next Planned Task (Day 12):** Multimodal extraction — visual encoding / multimodal embeddings (**Planned, Not Started**).
  - *Do not start Day 12 implementation until explicitly instructed.*
  - **Roadmap Context (Days 8–14):** Extracting images, figures, and tables; visual encoding/multimodal embeddings; table parsing; multimodal document representation.
  - **Day 12 Boundaries:**
    - Do not implement any Day 12 functionality now.
    - Do not add unnecessary frameworks or features.
    - Do not alter the established roadmap or mark future days complete.





