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
| **Image Processing** | Pillow (`pillow==12.3.0`) | Verified (Installed in `.venv`) |
| **LLM Inference** | LLM Provider or Local Alternative | Planned (Unverified) |
| **Evaluation** | Ragas | Planned (Unverified) |
| **Observability** | Langfuse | Planned (Unverified) |

---

## Current State

- **Status:** Day 14 Complete
- **Date:** October 17, 2026
- **Environment:** Windows, Python 3.13.5, active `.venv`
- **Version Control:** Git initialized locally; no commits made (files uncommitted/staged as requested)
- **Configuration & Dependencies:**
  - `requirements.txt`: Pinned to `pymupdf==1.28.2`, `pytest==9.1.1`, `sentence-transformers==6.1.0`, `faiss-cpu==1.15.1`, and `pillow==12.3.0`
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


### Day 12 — October 16, 2026 (Completed)
- **Completed Tasks:**
  - **Task 1: Unified Multimodal Element Representation (`app/multimodal/representation.py`)**
    - Created explicit `ElementType` str-backed Enum supporting all scientific-paper modalities: `text`, `image`, `caption`, `table`, and `equation` (with automatic synonym mapping for `figure` and `formula`).
    - Implemented `MultimodalElement` dataclass preserving exact provenance (`source_filename`, 1-based `page_number >= 1`, `element_id`), modality-aware `content` (strings for text/table/equation/caption, raw binary bytes or path references for images), vector `embedding` slot, `related_element_ids` relationship links, and modality-specific `metadata`.
    - Added dedicated adapter functions without modifying existing extractors: `from_text_chunk`, `from_image`, `from_caption`, `from_table`, `from_equation`, as well as universal `to_multimodal_element` and classmethod adapters.
    - Preserved Day 11 `CaptionMatch` relationships via bidirectional `attach_caption_matches` and direct `matches` injection.
    - Added 24 unit tests in `tests/test_representation.py`.
  - **Task 2: Modality-Aware Multimodal Embedding Layer (`app/multimodal/embedding.py`)**
    - Created abstract interfaces `BaseVisualEmbedder` and `BaseMultimodalEmbedder`.
    - Reused existing `SentenceTransformerEmbedder` (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions) for text-like elements: plain text chunks, caption text, Markdown formatted tables, and LaTeX equations.
    - Implemented `SentenceTransformerVisualEmbedder` using local pretrained `sentence-transformers/clip-ViT-B-32` (512 dimensions) supporting image bytes, file paths, and PIL Images with full vector validation (finite float32, dimensionality, count).
    - Designed `MultimodalEmbedder` to handle mixed-modality batches: elements are partitioned by modality, dispatched to modality-appropriate encoders in batches, and reconstructed into their original input sequence preserving 1:1 element mapping.
    - Built `MultimodalEmbeddedElement` and `MultimodalEmbeddingResult` container objects with type-filtered vector extraction (`get_vectors_by_type`) and strict dimensional safety guards (`to_numpy_matrix`).
    - Added 22 unit tests in `tests/test_multimodal_embedding.py`.
  - **Task 3: Multimodal Collection / Assembly and Integration (`app/multimodal/collection.py`)**
    - Created `MultimodalDocument` (aliased as `MultimodalCollection`) representing a scientific research paper as an indexed collection of unified multimodal elements.
    - Implemented element access and graph traversal methods: `get_by_id`, `get_by_type`, `get_by_page`, `get_related`, `get_embedding`, `embedded_elements()`, and `get_vectors_by_type()`.
    - Implemented `assemble_multimodal_document()` (and alias `assemble_multimodal_collection()`) accepting heterogeneous extracted items (`text_chunks`, `images`, `captions`, `tables`, `equations`, `caption_matches`) and producing a unified document while validating element ID uniqueness and establishing bidirectional links (image <-> caption, table <-> caption).
    - Integrated with `MultimodalEmbedder` via `doc.embed()` to associate vectors with elements while keeping vector spaces strictly separated (384-d text vs 512-d image).
    - Added 6 focused integration tests in `tests/test_multimodal_collection.py`.
- **Changed Files:**
  - `app/multimodal/representation.py` (Created)
  - `app/multimodal/embedding.py` (Created)
  - `app/multimodal/collection.py` (Created)
  - `app/multimodal/__init__.py` (Modified)
  - `requirements.txt` (Modified — added `pillow==12.3.0`)
  - `tests/test_representation.py` (Created)
  - `tests/test_multimodal_embedding.py` (Created)
  - `tests/test_multimodal_collection.py` (Created)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\.venv\Scripts\python -m pytest tests/test_representation.py -v`: **24 passed in 0.41s**
  - `.\.venv\Scripts\python -m pytest tests/test_multimodal_embedding.py -v`: **22 passed in 0.69s**
  - `.\.venv\Scripts\python -m pytest tests/test_multimodal_collection.py -v`: **6 passed in 0.43s**
  - `.\.venv\Scripts\python -m pytest -q`: **188 passed in 16.80s** (all 136 baseline tests + 52 new Day 12 tests)
- **Decisions Made:**
  - **Separate Embedding Spaces:** Kept text-like embeddings (384-d via `all-MiniLM-L6-v2`) and visual embeddings (512-d via `clip-ViT-B-32`) strictly separated without artificial concatenation or premature projection layers. Downstream retrieval (Days 13-14) will manage modality-specific indices.
  - **Modality-Aware Content:** Text, table, equation, and caption representations embed their meaningful textual representations; images embed raw bytes or image references.
  - **Exact Provenance & Identity:** Enforced strict 1-based page numbers (`>= 1`), non-empty strings for IDs/filenames, and uniqueness validation in `MultimodalDocument`.
  - **Relationship Preservation:** Reused Day 11 `CaptionMatch` structures without modifications, enabling bidirectional traversals (`doc.get_related("img_id")` -> `[cap_elem]`, `doc.get_related("cap_id")` -> `[img_elem]`, `doc.get_related("tbl_id")` -> `[cap_elem]`).
  - **Test Isolation:** Used mock embedders in unit/integration tests to ensure the entire test suite executes in ~16s without downloading multi-hundred-megabyte weights on every run.
- **Remaining Limitations:**
  - FAISS index currently only indexes text chunks; multimodal vector indexing and multimodal retrieval remain to be implemented in Days 13–14.
  - Cross-modal queries (e.g. text query matching an image directly) require either shared-space CLIP text encoding or separate multimodal index partitions.
- **Blockers:**
  - None.


### Day 13 — October 16, 2026 (Completed)
- **Completed Tasks:**
  - **Task 1: Modality-Specific Multimodal FAISS Indexing (`app/indexing/multimodal_models.py`, `app/indexing/multimodal_faiss_index.py`)**
    - Created `MultimodalSearchResult` dataclass in `app/indexing/multimodal_models.py`: wraps the original `MultimodalElement` and exposes first-class accessors for `score`, `element_id`, `element_type`, `source_filename`, 1-based `page_number`, `related_element_ids`, `content`, and `metadata` without duplicating stored data.
    - Implemented `ModalityFAISSIndex`: single-modality `faiss.IndexFlatIP` wrapper with L2-normalization for cosine similarity, fixed-dimensionality enforcement, rejection of dimension mismatches and non-finite vectors (NaN/Inf), bidirectional `int → MultimodalElement` and `element_id → int` mappings, and safe empty-index handling.
    - Implemented `MultimodalFAISSIndex`: high-level composite index maintaining two strictly separate sub-indices — `text_index` (384-dimensional) for `ElementType.TEXT`, `CAPTION`, `TABLE`, and `EQUATION` elements, and `visual_index` (512-dimensional) for `ElementType.IMAGE` elements. Vectors are never concatenated and no projection layer is introduced.
    - Added `add_embedding_result(MultimodalEmbeddingResult)` and `add_document(MultimodalDocument)` ingestion methods; classmethod factories `from_embedding_result` and `from_document`; and `build_multimodal_index()` convenience function.
    - Added modality-dispatched search methods: `search_text_by_vector`, `search_visual_by_vector`, `search_text`, and `search_visual`.
    - Updated `app/indexing/__init__.py` to re-export `ModalityFAISSIndex`, `MultimodalFAISSIndex`, `MultimodalSearchResult`, and `build_multimodal_index`. The existing `FAISSIndex` and `SearchResult` remain completely unchanged.
    - Added 18 focused tests in `tests/test_multimodal_indexing.py` across categories: dimensions, dimension safety (cross-modal rejection), modality routing, FAISS ID mapping, retrieval and ranking, provenance, validation (NaN/Inf/top_k), and a full mixed-document integration test.
  - **Task 2: Modality-Aware Multimodal Retrieval (`app/retrieval/multimodal_pipeline.py`)**
    - Created `MultimodalRetrievalPipeline` in `app/retrieval/multimodal_pipeline.py`: orchestrates similarity search over `MultimodalFAISSIndex` with strict modality isolation — `retrieve_text` / `search_text` queries only the 384-d text index; `retrieve_visual` / `search_visual` queries only the 512-d visual index.
    - Added a generic `retrieve(query, modality, top_k)` dispatcher accepting `'text'` or `'visual'` modality strings for convenient programmatic use.
    - Reused the existing `BaseEmbedder` / `SentenceTransformerEmbedder` infrastructure for text query encoding and `BaseVisualEmbedder` / `SentenceTransformerVisualEmbedder` for image query encoding; no new models introduced.
    - All returned `MultimodalSearchResult` objects preserve the original `MultimodalElement` along with exact `element_id`, `element_type`, `source_filename`, 1-based `page_number`, similarity `score`, `related_element_ids`, `content`, and `metadata`.
    - Added `build_multimodal_pipeline()` convenience factory.
    - Updated `app/retrieval/__init__.py` to re-export `MultimodalRetrievalPipeline` and `build_multimodal_pipeline`. The existing text-only `RetrievalPipeline` is completely unchanged.
    - Added 13 focused tests in `tests/test_multimodal_retrieval.py` across categories: text retrieval (vector and string query, ranking), visual retrieval (vector and bytes query), modality isolation (text results never contain images; visual results never contain text-like elements), provenance integrity, image↔caption relationship traversal via `MultimodalDocument.get_related()`, and validation (empty query, invalid `top_k`, dimension mismatch, invalid modality dispatch, pipeline init validation).
- **Changed Files:**
  - `app/indexing/multimodal_models.py` (Created)
  - `app/indexing/multimodal_faiss_index.py` (Created)
  - `app/indexing/__init__.py` (Modified)
  - `app/retrieval/multimodal_pipeline.py` (Created)
  - `app/retrieval/__init__.py` (Modified)
  - `tests/test_multimodal_indexing.py` (Created)
  - `tests/test_multimodal_retrieval.py` (Created)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\.venv\Scripts\python -m pytest tests/test_multimodal_indexing.py -v`: **18 passed in 0.65s**
  - `.\.venv\Scripts\python -m pytest tests/test_multimodal_retrieval.py -v`: **13 passed in 0.60s**
  - `.\.venv\Scripts\python -m pytest -q`: **219 passed in 16.29s** (all 188 baseline + 18 Day 13 indexing + 13 Day 13 retrieval)
- **Decisions Made:**
  - **Strict Vector Space Separation:** Text-like embeddings (384-d) and visual embeddings (512-d) remain in completely separate FAISS indices. No concatenation, no projection layer, no combined index.
  - **Reuse over Duplication:** `ModalityFAISSIndex` reuses the same L2-normalization and `IndexFlatIP` pattern as the existing `FAISSIndex`. The existing text-only pipeline is untouched.
  - **Provenance Preserved End-to-End:** FAISS integer IDs map directly to the original `MultimodalElement` instances, preserving all provenance fields and relationship graphs through to `MultimodalSearchResult`.
  - **Cross-Modal Retrieval Deferred:** A text query does not search the visual index and vice versa; cross-modal alignment (e.g. CLIP shared space) is intentionally left for a later day.
  - **Backward Compatibility:** Existing `FAISSIndex`, `SearchResult`, and `RetrievalPipeline` are fully intact; all 188 pre-Day-13 tests continue to pass without modification.
- **Remaining Limitations:**
  - Cross-modal retrieval (text query → image results via shared CLIP space) is not yet implemented.
  - Sparse BM25 retrieval and hybrid dense+sparse fusion (RRF) are deferred to Days 15–17.
  - Cross-encoder reranking is deferred to Days 15–17.
  - Vector index persistence (save/load) is not yet implemented.
  - LLM generation integration and UI remain part of later work.
- **Blockers:**
  - None.



### Day 14 — October 17, 2026 (Completed)
- **Completed Tasks:**
  - **Step 1: Multimodal Retrieved-Context Modeling and Assembly (`app/retrieval/multimodal_models.py`, `app/retrieval/multimodal_assembler.py`)**
    - Created `MultimodalCitation` dataclass in `app/retrieval/multimodal_models.py`: captures `citation_label`, `element_id`, `element_type`, `source_filename`, 1-based `page_number`, and `score`; includes `__post_init__` validation (non-empty fields, page ≥ 1).
    - Created `MultimodalRetrievedContext` dataclass: holds `formatted_context`, `citations`, and `search_results`; provides helper properties — `elements`, `visual_results`, `text_results`, `related_element_ids` — and lookup methods `get_citation_by_label()` and `get_result_by_element_id()`.
    - Implemented `MultimodalContextAssembler`: converts `List[MultimodalSearchResult]` into `MultimodalRetrievedContext` with sequential `[1]`, `[2]`, ... citation markers, source filename, and 1-based page references.
    - Modality-specific evidence formatting: `TEXT` → readable text; `CAPTION` → caption content; `TABLE` → table markdown; `EQUATION` → formula/LaTeX string; `IMAGE` → safe text descriptor (format, dimensions) — raw image bytes are never emitted into formatted prompt text.
    - Relationship resolution: if an image result has `related_element_ids` and a `MultimodalDocument` is provided, the linked caption text is automatically resolved and included in the formatted context. Without a document, related IDs are exposed as a reference line. No relationships are invented.
    - Added `assemble_multimodal_context()` convenience function.
    - Updated `app/retrieval/__init__.py` to re-export `MultimodalCitation`, `MultimodalRetrievedContext`, `MultimodalContextAssembler`, and `assemble_multimodal_context`.
    - Added 12 focused unit tests in `tests/test_multimodal_assembly.py` covering all 5 element type formats, sequential citation labels, source filename and page preservation, relationship/caption exposure (with and without document), empty results, original search result object preservation, helper property behavior, and `MultimodalCitation` validation.
  - **Step 2: `retrieve_and_assemble()` Integration into `MultimodalRetrievalPipeline` (`app/retrieval/multimodal_pipeline.py`)**
    - Extended `MultimodalRetrievalPipeline.__init__()` to accept an optional `assembler: MultimodalContextAssembler` parameter (defaults to a fresh `MultimodalContextAssembler()`).
    - Added `retrieve_and_assemble(query, modality, top_k, document)`: calls the appropriate existing retrieval method (`retrieve_text` or `retrieve_visual`) and passes results to the assembler, returning `MultimodalRetrievedContext`. Supports `modality='text'` and `modality='visual'`.
    - Updated `build_multimodal_pipeline()` to accept and pass through an optional `assembler` argument.
    - All existing `retrieve_text`, `retrieve_visual`, and `retrieve` methods remain completely unchanged.
    - Added 7 focused integration tests in `tests/test_multimodal_retrieval.py` (Category G: `TestRetrieveAndAssemble`) covering: text query assembly, visual query assembly, image→caption resolution when a document is supplied, `top_k` propagation, invalid modality handling, empty retrieval results, and backward compatibility of existing retrieval methods.
  - **Step 3: End-to-End Multimodal Integration Tests (`tests/test_multimodal_integration.py`)**
    - Created a complete deterministic integration test suite proving the following flow works without any real model weights or external API calls:
      `MultimodalDocument` → `MultimodalEmbedder` (384-d text / 512-d visual) → `MultimodalFAISSIndex` → `MultimodalRetrievalPipeline.retrieve_and_assemble()` → `MultimodalRetrievedContext` (formatted grounded context + citations + original results).
    - Test fixture assembles a realistic 5-element scientific paper: `TEXT` (abstract), `IMAGE` (architecture diagram with width/height metadata), `CAPTION` (linked to image via `CaptionMatch`), `TABLE` (BLEU results markdown), and `EQUATION` (scaled dot-product attention formula).
    - Uses lightweight `DeterministicTextEmbedder` (384-d) and `DeterministicVisualEmbedder` (512-d) mapping semantic concepts to orthogonal unit-vector dimensions — no model download required.
    - Added 5 end-to-end tests: (1) embedding and modality partitioning (4 elements to text_index, 1 to visual_index); (2) text query → `TEXT` element hit with full citation provenance; (3) `TABLE` and `EQUATION` queries each retrieve their respective elements with correct page numbers and formatted content; (4) visual query → `IMAGE` hit with score ≈ 1.0, linked caption resolved from `MultimodalDocument`, raw bytes confirmed absent from formatted context; (5) strict modality isolation across the complete pipeline (text retrieval never returns images; visual retrieval never returns text-like elements).
- **Architecture Preserved:**
  - The 384-d text and 512-d visual FAISS vector spaces remain strictly separate throughout — no vector concatenation, no projection layer, no combined index.
  - Image→caption resolution uses existing `MultimodalDocument.get_by_id()` traversal; no new cross-modal embedding infrastructure was introduced.
  - Backward compatibility: all 219 pre-Day-14 tests continue to pass without modification.
- **Changed Files:**
  - `app/retrieval/multimodal_models.py` (Created)
  - `app/retrieval/multimodal_assembler.py` (Created)
  - `app/retrieval/multimodal_pipeline.py` (Modified — added `retrieve_and_assemble()` and `assembler` parameter)
  - `app/retrieval/__init__.py` (Modified — added re-exports for Day 14 symbols)
  - `tests/test_multimodal_assembly.py` (Created)
  - `tests/test_multimodal_retrieval.py` (Modified — added Category G tests)
  - `tests/test_multimodal_integration.py` (Created)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\.venv\Scripts\python -m pytest tests/test_multimodal_assembly.py -v`: **12 passed in 3.43s**
  - `.\.venv\Scripts\python -m pytest tests/test_multimodal_retrieval.py -v`: **20 passed in 0.77s** (13 baseline + 7 new Day 14)
  - `.\.venv\Scripts\python -m pytest tests/test_multimodal_assembly.py tests/test_multimodal_retrieval.py -v`: **32 passed in 0.58s**
  - `.\.venv\Scripts\python -m pytest tests/test_multimodal_integration.py -v`: **5 passed in 0.68s**
  - `.\.venv\Scripts\python -m pytest -q`: **243 passed in 15.68s** (all 219 baseline + 12 assembly + 7 pipeline integration + 5 end-to-end integration), **0 failures**
- **Decisions Made:**
  - **No Architecture Changes:** Strict modality isolation (384-d text / 512-d visual) is preserved throughout. No concatenation, projection, hybrid fusion, or cross-modal alignment was introduced.
  - **Image Safety:** Raw image bytes are never emitted into the formatted context string; images are represented as structured text descriptors.
  - **Relationship Resolution via Document:** Caption resolution for image hits uses the passed-in `MultimodalDocument` reference, not a new dedicated resolver. No relationships are invented — only those already encoded in `related_element_ids` are used.
  - **Assembler as Optional Parameter:** The assembler is injected into `MultimodalRetrievalPipeline` as an optional dependency, keeping it swappable and testable in isolation.
  - **Backward Compatibility:** `retrieve_text`, `retrieve_visual`, and `retrieve` remain untouched. `build_multimodal_pipeline` gains only an optional `assembler` kwarg; existing call sites require no changes.
- **Remaining Limitations:**
  - Cross-modal retrieval (text query → image results via shared CLIP space) is not yet implemented.
  - Sparse BM25 retrieval and hybrid dense+sparse fusion (RRF) are deferred to Days 15–17.
  - Cross-encoder reranking is deferred to Days 15–17.
  - Vector index persistence (save/load) is not yet implemented.
  - LLM generation integration and UI remain part of later work.
- **Blockers:**
  - None.

---

## Day 15: Hybrid Retrieval — Task 1: Sparse BM25 Retrieval Foundation

- **Status:** Complete
- **Goal:** Implement the sparse BM25 retrieval foundation operating over text-like multimodal elements (TEXT, CAPTION, TABLE, EQUATION) with strict IMAGE exclusion, deterministic lightweight tokenization, provenance preservation, and standard BM25 scoring.
- **Components Built:**
  - **Deterministic Tokenization (`app/indexing/bm25_index.py`):**
    - `tokenize(text: str) -> List[str]`: Lightweight regex-based tokenizer normalizing case and extracting alphanumeric tokens while preserving hyphenated and underscore-linked terms (`d_k`, `w_q`, `bleu-4`, `transformer-base`).
    - Strips markdown table syntax, LaTeX math decorators, and punctuation without requiring heavyweight external NLP dependencies.
  - **In-Memory BM25 Index (`app/indexing/bm25_index.py`):**
    - `BM25Index`: Okapi BM25 index with configurable $k_1$ (default 1.5, non-negative) and $b$ (default 0.75, $0.0 \le b \le 1.0$).
    - Operates strictly on text-like element types: `ElementType.TEXT`, `ElementType.CAPTION`, `ElementType.TABLE`, `ElementType.EQUATION`.
    - IMAGE elements (`ElementType.IMAGE`) are strictly excluded and never enter the BM25 corpus.
    - Preserves full original `MultimodalElement` objects with `element_id`, `element_type`, `source_filename`, 1-based `page_number`, `metadata`, `related_element_ids`, and `content`.
    - Returns standard `MultimodalSearchResult` objects with positive float BM25 scores, maintaining seamless downstream compatibility with the multimodal assembler and pipeline.
    - Supports `add_elements()`, `add_document()`, `search()`, `clear()`, `from_elements()`, `from_document()`, and `build_bm25_index()`.
    - Handles empty index, empty/whitespace queries, non-positive `top_k`, non-matching queries, and returns fewer results when matches < `top_k`.
  - **Re-exports (`app/indexing/__init__.py`):**
    - Exposed `BM25Index`, `build_bm25_index`, and `tokenize` alongside existing FAISS and Multimodal indexing symbols without altering dense retrieval behavior.
  - **Comprehensive Unit Tests (`tests/test_bm25_index.py`):**
    - 38 focused unit tests across 8 categories:
      1. Deterministic tokenization (case normalization, punctuation, scientific subscripts, hyphens, LaTeX equations, markdown tables, empty/invalid inputs).
      2. Modality filtering & strict IMAGE exclusion (all text-like modalities indexed, images strictly rejected).
      3. Parameter validation ($k_1 < 0$, $b < 0$ or $b > 1$, non-numeric types, non-positive `top_k`, invalid element types, duplicate element ID rejection).
      4. Empty and edge cases (empty index, empty/whitespace queries, invalid query type, no matches, fewer results than `top_k`, index clearing).
      5. BM25 ranking (TF sensitivity, length normalization via $b$, multi-term query ranking, equation retrieval on `d_k`, table retrieval on `bleu-4`).
      6. Provenance preservation (`MultimodalSearchResult` fields, original element reference, metadata, relationships, positive scores).
      7. `MultimodalDocument` integration & factory constructors (`from_document`, `from_elements`, `build_bm25_index`).
      8. Non-regression verification confirming `FAISSIndex` and `MultimodalFAISSIndex` components remain completely untouched.
- **Changed Files:**
  - `app/indexing/bm25_index.py` (Created)
  - `app/indexing/__init__.py` (Modified — added BM25 symbols)
  - `tests/test_bm25_index.py` (Created — 38 tests)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\.venv\Scripts\python -m pytest tests/test_bm25_index.py -v`: **38 passed in 1.67s**
  - `.\.venv\Scripts\python -m pytest tests/test_multimodal_indexing.py -v`: **18 passed in 0.80s** (confirmed 0 regression to dense indexing)
- **Decisions Made:**
  - **Result Format Compatibility:** Reused `MultimodalSearchResult` from `app/indexing/multimodal_models.py` rather than creating a parallel result model, ensuring zero impedance mismatch with `MultimodalContextAssembler` and future hybrid fusion layers.
  - **Robertson-Spärck Jones / Lucene IDF:** Used $\ln(1.0 + (N - n_q + 0.5) / (n_q + 0.5))$ to ensure IDF is non-negative and prevent negative scores when term frequency in corpus exceeds 50%.
  - **Strict Modality Isolation:** IMAGE elements are filtered out prior to index insertion and never enter the BM25 inverted index.
  - **Lightweight Tokenizer:** Implemented with standard library `re` without introducing external dependencies (such as NLTK or spaCy), preserving scientific tokens like `d_k` and `bleu-4`.
- **Remaining Limitations:**
  - Hybrid dense+sparse fusion (RRF) is deferred to subsequent Day 15 tasks.
  - Cross-encoder reranking is deferred to Day 16.
  - Vector index persistence (save/load) is not yet implemented.
  - LLM generation integration and UI remain part of later work.
- **Blockers:**
  - None.

---

## Day 15: Hybrid Retrieval — Task 2: BM25 Sparse Retriever Integration

- **Status:** Complete
- **Goal:** Integrate the completed BM25 sparse retriever into `MultimodalRetrievalPipeline` via an optional dependency, exposing a dedicated `retrieve_sparse()` method while preserving existing dense retrieval behavior.
- **Components Built:**
  - **`MultimodalRetrievalPipeline` Sparse Integration (`app/retrieval/multimodal_pipeline.py`):**
    - Extended `__init__()` to accept an optional `bm25_index: Optional[BM25Index] = None` dependency with runtime type validation (`isinstance(bm25_index, BM25Index)`).
    - Added `has_bm25` property indicating whether sparse retrieval is available.
    - Implemented `retrieve_sparse(query: str, top_k: Optional[int] = None) -> List[MultimodalSearchResult]`:
      - Dispatches sparse queries to `self.bm25_index.search(query, top_k=effective_k)`.
      - Enforces strict parameter validation: raises `RuntimeError` if BM25 is not configured, `ValueError` if `top_k <= 0`, `TypeError` if `query` is not a string.
      - Returns `[]` for empty or whitespace query strings.
      - Returns standard `List[MultimodalSearchResult]` with positive BM25 float scores.
    - Added `search_sparse = retrieve_sparse` convenience alias matching `search_text` and `search_visual`.
    - Extended `retrieve(query, modality="sparse"|"bm25")` to route queries to `retrieve_sparse`.
    - Updated `build_multimodal_pipeline()` factory with optional `bm25_index` pass-through.
    - Preserved exact dense retrieval behavior: `retrieve_text()`, `retrieve_visual()`, `retrieve()`, and `retrieve_and_assemble()` remain 100% backward-compatible.
  - **Comprehensive Integration Tests (`tests/test_multimodal_sparse_retrieval.py`):**
    - 21 focused unit tests across 6 categories:
      1. Sparse retrieval delegation (`retrieve_sparse`, `search_sparse` alias, `retrieve(modality='sparse')`, `retrieve(modality='bm25')`).
      2. Modality coverage (successful retrieval of `TEXT`, `CAPTION`, `TABLE`, and `EQUATION`).
      3. Strict `IMAGE` exclusion (confirmed images never returned by sparse retrieval).
      4. Parameter propagation and provenance preservation (`top_k`, default fallback, all metadata and provenance fields preserved).
      5. Empty & edge cases (empty string, whitespace, non-matching terms, empty BM25 index).
      6. Unconfigured BM25 & validations (`RuntimeError` on unconfigured pipeline, `ValueError` on non-positive `top_k`, `TypeError` on non-string queries or non-BM25 index instances).
      7. Backward compatibility (dense text retrieval, dense visual retrieval, and prompt-grounded context assembly remain untouched).
- **Changed Files:**
  - `app/retrieval/multimodal_pipeline.py` (Modified)
  - `tests/test_multimodal_sparse_retrieval.py` (Created — 21 tests)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\.venv\Scripts\python -m pytest tests/test_multimodal_sparse_retrieval.py -v`: **21 passed in 0.65s**
  - `.\.venv\Scripts\python -m pytest tests/test_multimodal_retrieval.py tests/test_bm25_index.py tests/test_multimodal_integration.py -v`: **63 passed in 0.75s**
- **Decisions Made:**
  - **Fail Fast for Unconfigured BM25:** Calling `retrieve_sparse()` on a pipeline without a configured `BM25Index` raises `RuntimeError` rather than silently returning `[]`, allowing callers and future hybrid fusion layers to distinguish between missing configuration vs zero matching hits.
  - **Modality Dispatch Integration:** Added `'sparse'` and `'bm25'` to `retrieve()` so generic callers can dispatch across dense text, dense visual, and sparse text through a single method.
  - **Full Backward Compatibility:** Kept all existing constructor signatures and default arguments backwards-compatible; pipelines without a BM25 index work identically to pre-Day-15 behavior.
- **Remaining Limitations:**
  - Hybrid dense+sparse fusion (RRF) is deferred to Task 3.
  - Cross-encoder reranking is deferred to Day 16.
  - Vector index persistence (save/load) is not yet implemented.
  - LLM generation integration and UI remain part of later work.
- **Blockers:**
  - None.

---

## Day 15: Hybrid Retrieval — Task 3: Reciprocal Rank Fusion (RRF) Standalone Component

- **Status:** Complete
- **Goal:** Implement standard Reciprocal Rank Fusion (RRF) as a standalone, independent retrieval-fusion component capable of combining multiple ranked result lists (e.g. dense FAISS and sparse BM25) with provenance preservation, deduplication, and deterministic tie-breaking.
- **Components Built:**
  - **Standalone RRF Component (`app/retrieval/rrf.py`):**
    - Implemented `ReciprocalRankFusion` class and `reciprocal_rank_fusion()` convenience function.
    - Standard formula: $\text{RRF\_score}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}(d, r)}$ with 1-based ranks.
    - Default smoothing constant $k = 60$ (`DEFAULT_RRF_K`), configurable and validated ($k > 0$).
    - Deduplicates items across multiple ranked lists by unique `element_id`.
    - Handles intra-list duplicates by preserving the best (first) rank encountered.
    - Combines reciprocal rank contributions for overlapping elements; preserves single-list contributions for dense-only or sparse-only elements.
    - Replaces original vector similarity / BM25 scores with pure combined RRF scores.
    - Preserves complete `MultimodalSearchResult` provenance (`element`, `element_id`, `element_type`, `source_filename`, `page_number`, `metadata`, `related_element_ids`, `content`).
    - Sorts descending by RRF score, breaking ties deterministically by `element_id`.
    - Supports optional `top_k` parameter with positive integer validation.
    - Gracefully handles empty result lists and empty sublists.
  - **Re-exports (`app/retrieval/__init__.py`):**
    - Re-exported `DEFAULT_RRF_K`, `ReciprocalRankFusion`, and `reciprocal_rank_fusion`.
  - **Comprehensive Unit Tests (`tests/test_rrf.py`):**
    - 23 focused unit tests across 6 categories:
      1. Formula & rank contribution ($1 / (k + rank)$, default $k=60$, custom $k=10$, overlapping results combination).
      2. Deduplication and single-list survival (inter-list deduplication, intra-list best-rank deduplication, dense-only survival, sparse-only survival).
      3. Provenance and score replacement (original FAISS/BM25 scores replaced with pure RRF scores, full `MultimodalSearchResult` fields intact).
      4. Deterministic ordering and tie-breaking (descending order, alphabetical `element_id` tie-breaking).
      5. Empty inputs and edge cases (empty lists, all-empty sublists, mixed empty/populated lists, `top_k` limiting, fewer results than `top_k`).
      6. Parameter validation (invalid $k \le 0$, non-numeric $k$, invalid `top_k`, non-sequence/malformed lists, non-`MultimodalSearchResult` items).
- **Changed Files:**
  - `app/retrieval/rrf.py` (Created — 144 lines)
  - `app/retrieval/__init__.py` (Modified — added RRF re-exports)
  - `tests/test_rrf.py` (Created — 23 tests)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\.venv\Scripts\python -m pytest tests/test_rrf.py -v`: **23 passed in 0.89s**
  - `.\.venv\Scripts\python -m pytest tests/test_rrf.py tests/test_multimodal_sparse_retrieval.py tests/test_multimodal_retrieval.py -v`: **64 passed in 0.64s**
- **Decisions Made:**
  - **Isolated from Execution:** Kept RRF purely functional and independent of FAISS or BM25 index instances so it operates deterministically on arbitrary ranked lists of `MultimodalSearchResult`.
  - **Score Replacement:** Original similarity/BM25 scores are replaced with pure RRF scores rather than blending heterogeneous score scales, preventing unnormalized score biases.
  - **Intra-list Best Rank:** If an element appears multiple times within a single list, only its best rank is counted to prevent duplicate self-boosting.
- **Remaining Limitations:**
  - Full hybrid retrieval pipeline integrating dense FAISS + sparse BM25 + RRF is the next planned step.
  - Cross-encoder reranking is deferred to Day 16.
  - Vector index persistence (save/load) is not yet implemented.
  - LLM generation integration and UI remain part of later work.
- **Blockers:**
  - None.

---

## Day 15: Hybrid Retrieval — Task 4: Hybrid Dense + Sparse + RRF Integration

- **Status:** Complete
- **Goal:** Integrate dense FAISS (384-d), sparse BM25, and Reciprocal Rank Fusion (RRF) into `MultimodalRetrievalPipeline`, providing a dedicated `hybrid_retrieve_text()` method and `hybrid_retrieve_and_assemble()` prompt context assembler while strictly maintaining modality isolation and 100% backward compatibility.
- **Components Built:**
  - **Hybrid Retrieval Integration (`app/retrieval/multimodal_pipeline.py`):**
    - Implemented `hybrid_retrieve_text(query, dense_top_k, sparse_top_k, top_k, rrf_k)`:
      - Validates query string, positive top_k thresholds, and positive RRF $k$.
      - Raises clear `RuntimeError("BM25 index is not configured...")` if called without a configured `BM25Index`.
      - Dispatches dense retrieval to `self.retrieve_text(query, top_k=effective_dense_k)`.
      - Dispatches sparse retrieval to `self.retrieve_sparse(query, top_k=effective_sparse_k)`.
      - Fuses ranked lists using `reciprocal_rank_fusion([dense_results, sparse_results], k=rrf_k, top_k=effective_top_k)`.
      - Returns `List[MultimodalSearchResult]` ordered by descending combined RRF score with deterministic tie-breaking.
      - Enforces strict modality isolation: operates only on text-like elements (`TEXT`, `CAPTION`, `TABLE`, `EQUATION`); `IMAGE` elements are strictly excluded.
      - Preserves full provenance (`element_id`, `page_number`, `metadata`, `related_element_ids`, `content`, `source_filename`).
    - Implemented `hybrid_retrieve_and_assemble()` leveraging existing `MultimodalContextAssembler` for structured citation generation (`[1]`, `[2]`, ...).
    - Added convenience aliases: `search_hybrid_text = hybrid_retrieve_text`, `retrieve_hybrid_text = hybrid_retrieve_text`.
    - Extended `retrieve(query, modality="hybrid"|"hybrid_text"|"dense_sparse")` to route queries to `hybrid_retrieve_text`.
    - Preserved exact dense retrieval behavior: `retrieve_text()`, `retrieve_sparse()`, `retrieve_visual()`, `retrieve()`, and `retrieve_and_assemble()` remain 100% backward-compatible.
  - **Comprehensive Integration Tests (`tests/test_multimodal_hybrid_retrieval.py`):**
    - 19 focused integration tests covering:
      - Overlapping elements receive combined RRF scores ($1/(60+1) + 1/(60+2)$).
      - Dense-only results survive.
      - Sparse-only results survive.
      - Final `top_k` strictly respected.
      - Strict modality isolation: `IMAGE` elements never appear in hybrid text results.
      - Full provenance preservation across FAISS → BM25 → RRF → output.
      - Empty and edge cases (both empty, dense populated + sparse empty, empty string query).
      - Clear `RuntimeError` when BM25 is missing.
      - Input validation (`top_k <= 0`, `dense_top_k <= 0`, `sparse_top_k <= 0`, non-string query, invalid `rrf_k`).
      - Full backward compatibility for dense text, sparse, visual, and prompt-ready context assembly.
- **Changed Files:**
  - `app/retrieval/multimodal_pipeline.py` (Modified)
  - `tests/test_multimodal_hybrid_retrieval.py` (Created — 19 tests)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\.venv\Scripts\python -m pytest tests/test_multimodal_hybrid_retrieval.py -v`: **19 passed in 0.81s**
  - `.\.venv\Scripts\python -m pytest tests/test_multimodal_hybrid_retrieval.py tests/test_rrf.py tests/test_multimodal_sparse_retrieval.py tests/test_multimodal_retrieval.py -v`: **83 passed in 0.72s**
- **Decisions Made:**
  - **Target Flow Exact Alignment:** Dense text candidates (384-d FAISS) and sparse text candidates (BM25) are retrieved in parallel and combined via standard RRF with $k=60$. Original vector similarity and BM25 scores are replaced with pure RRF scores to avoid disparate score scale distortions.
  - **Modality Isolation Preserved:** Dense text and sparse BM25 indices independently exclude visual elements; hybrid retrieval is strictly text-like (`TEXT`, `CAPTION`, `TABLE`, `EQUATION`).
  - **Assembler Reuse:** Reused `MultimodalContextAssembler` without duplicating formatting or citation generation logic.
- **Remaining Limitations:**
  - Cross-encoder reranking is deferred to Day 16.
  - Vector index persistence (save/load) is not yet implemented.
  - LLM generation integration and UI remain part of later work.
- **Blockers:**
  - None.

---

## Day 16: Cross-Encoder Reranking — Task 1: Reranking Foundation

- **Status:** Complete
- **Goal:** Build an abstract `BaseReranker` interface and a concrete `CrossEncoderReranker` implementation that accepts a query and a list of `MultimodalSearchResult` candidates and returns them reordered by cross-encoder relevance score, without modifying any existing retrieval, BM25, RRF, or FAISS behavior.
- **Components Built:**
  - **Abstract Base Interface (`app/reranking/base.py`):**
    - `BaseReranker` ABC exposing `rerank(query, candidates, top_k) -> List[MultimodalSearchResult]`.
    - Enforces strict parameter validation in the abstract contract (non-empty string query, non-negative `top_k`).
  - **`CrossEncoderReranker` (`app/reranking/cross_encoder.py`):**
    - Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
    - Accepts an injectable `model` argument so unit tests run fully offline without downloading weights.
    - `default_candidate_text_extractor()` safely derives text from all element types including `IMAGE` metadata fallbacks.
    - Normalizes both 1D and 2D logit outputs from the cross-encoder; handles empty candidates and model score-count mismatches (`ScoreCountMismatchError`).
    - Deterministic tie-breaking on `(-score, element_id)`; preserves all original `MultimodalSearchResult` fields.
    - Configurable `batch_size`, `device`, and `max_length`.
  - **Re-exports (`app/reranking/__init__.py`):**
    - Exposed `BaseReranker`, `CrossEncoderReranker`, `ScoreCountMismatchError`, and `default_candidate_text_extractor`.
  - **Convenience shim (`app/retrieval/reranker.py`):**
    - Re-exports `BaseReranker` and `CrossEncoderReranker` for callers who import from the `retrieval` package.
  - **Comprehensive Unit Tests (`tests/test_reranker.py`):**
    - 40 focused unit tests covering: abstract interface enforcement, score-based reordering, top-k slicing, empty candidate handling, invalid query/top_k validation, score-count mismatch error, 1D/2D logit normalization, deterministic tie-breaking, and full provenance field preservation.
- **Changed Files:**
  - `app/reranking/__init__.py` (Created)
  - `app/reranking/base.py` (Created)
  - `app/reranking/cross_encoder.py` (Created)
  - `app/retrieval/reranker.py` (Created — re-export shim)
  - `tests/test_reranker.py` (Created — 40 tests)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\\.venv\\Scripts\\python -m pytest tests/test_reranker.py -v`: **40 passed**
- **Decisions Made:**
  - **Injectable Model:** The `model` constructor argument accepts any object exposing `.predict()`, making the reranker fully testable with deterministic fakes without network or disk access.
  - **Strict Isolation:** No existing BM25, RRF, FAISS, or pipeline code was modified; the reranker is a standalone component.
  - **Provenance Preservation:** Only the ordering and `score` field of returned `MultimodalSearchResult` objects change; all other fields (`element`, `element_id`, `element_type`, `source_filename`, `page_number`, `metadata`, `related_element_ids`, `content`) are passed through unmodified.
- **Remaining Limitations:**
  - Cross-encoder integration into `MultimodalRetrievalPipeline` is the next step.
  - Vector index persistence (save/load) is not yet implemented.
  - LLM generation integration and UI remain part of later work.
- **Blockers:**
  - None.

---

## Day 16: Cross-Encoder Reranking — Task 2: Pipeline Integration

- **Status:** Complete
- **Goal:** Integrate `CrossEncoderReranker` into `MultimodalRetrievalPipeline` as an optional injectable dependency, adding a `hybrid_retrieve_reranked()` method (Dense FAISS → BM25 → RRF → Cross-Encoder) while preserving 100% backward compatibility with all existing Day 15 retrieval methods.
- **Components Built:**
  - **`MultimodalRetrievalPipeline` Reranking Integration (`app/retrieval/multimodal_pipeline.py`):**
    - Extended `__init__()` to accept an optional `reranker: Optional[BaseReranker] = None` dependency.
    - Added `has_reranker` property.
    - Implemented `hybrid_retrieve_reranked(query, dense_top_k, sparse_top_k, candidate_top_k, top_k, rrf_k, reranker)`:
      - Validates query and all positive-integer thresholds.
      - Raises `RuntimeError` if neither the pipeline reranker nor an ad-hoc reranker is supplied.
      - Executes: dense FAISS retrieval → sparse BM25 retrieval → RRF candidate fusion → `CrossEncoderReranker.rerank()`.
      - Supports an ad-hoc `reranker` argument that overrides the pipeline-level reranker per call.
      - Preserves full `MultimodalSearchResult` provenance through the entire chain.
    - Implemented `hybrid_retrieve_reranked_and_assemble()` for end-to-end prompt-ready context with structured citations.
    - Added convenience aliases: `search_hybrid_reranked`, `retrieve_hybrid_reranked`, `reranked_hybrid_retrieve_text`, `reranked_hybrid_retrieve_and_assemble`.
    - Extended `retrieve(modality=\"hybrid_reranked\"|\"reranked_hybrid\"|\"hybrid_rerank\"|\"rerank\")` dispatch.
    - Preserved exact behavior of `retrieve_text()`, `retrieve_sparse()`, `retrieve_visual()`, `hybrid_retrieve_text()`, `retrieve_and_assemble()`, and all Day 15 methods.
  - **`app/retrieval/__init__.py`:** Re-exported `BaseReranker` and `CrossEncoderReranker` from the retrieval package.
  - **Comprehensive Integration Tests (`tests/test_multimodal_reranked_retrieval.py`):**
    - 22 focused integration tests covering: reranked pipeline end-to-end flow, top-k limiting after reranking, ad-hoc reranker override, missing reranker `RuntimeError`, candidate pool expansion before reranking, full provenance preservation, `hybrid_retrieve_reranked_and_assemble()` context assembly, modality dispatch via `retrieve()`, and full backward compatibility for all prior retrieval modes.
- **Changed Files:**
  - `app/retrieval/multimodal_pipeline.py` (Modified)
  - `app/retrieval/__init__.py` (Modified — added reranker re-exports)
  - `tests/test_multimodal_reranked_retrieval.py` (Created — 22 tests)
  - `PROJECT_PROGRESS.md` (Updated)
- **Commands & Actual Test Results:**
  - `.\\.venv\\Scripts\\python -m pytest tests/test_multimodal_reranked_retrieval.py -v`: **22 passed**
  - `.\\.venv\\Scripts\\python -m pytest tests/test_reranker.py tests/test_multimodal_reranked_retrieval.py tests/test_multimodal_hybrid_retrieval.py -v`: **81 passed**
- **Decisions Made:**
  - **Backward Compatibility First:** All existing constructor signatures and defaults remain unchanged; a pipeline without a reranker behaves identically to Day 15.
  - **Fail Fast:** `hybrid_retrieve_reranked()` raises `RuntimeError` immediately if no reranker is available rather than silently returning RRF-only results, preventing silent degradation.
  - **Reranker Reuse:** `CrossEncoderReranker` is consumed as-is; no reranking logic is duplicated inside the pipeline.
- **Remaining Limitations:**
  - Vector index persistence (save/load) is not yet implemented.
  - LLM generation integration and UI remain part of later work.
- **Blockers:**
  - None.

---

## Day 16: Cross-Encoder Reranking — Task 3: Live Evaluation on Sample Paper

- **Status:** Complete
- **Goal:** Evaluate whether Cross-Encoder reranking meaningfully improves retrieval quality over the Day 15 Hybrid RRF baseline using five representative queries against `data/sample_paper.pdf`.
- **Evaluation Approach:**
  - Built a full `MultimodalRetrievalPipeline` (384-d FAISS + BM25 + RRF + `CrossEncoderReranker`) from the canonical `data/sample_paper.pdf`.
  - For each query, compared side-by-side: Hybrid RRF top-5 results vs. Hybrid RRF + Cross-Encoder top-5 results.
  - Inspected ranking changes, element type (`TEXT`, `CAPTION`, `TABLE`, `EQUATION`, `IMAGE`), page number, element ID, and score.
- **Queries Evaluated:**
  1. Conceptual query: "attention mechanism scaled dot product"
  2. Keyword query: "multi-head attention"
  3. Equation/symbol query: "d_k scaling factor softmax"
  4. Architectural query: "encoder decoder layers"
  5. Figure query: "FIGURE 3 AI RMF NIST"
- **Key Findings:**
  - For most conceptual and keyword queries, Cross-Encoder reranking demonstrably improved precision: licensing/boilerplate chunks that keyword-matched surface terms were demoted, while substantive conceptual paragraphs were promoted.
  - The reranker successfully corrected keyword-overlap ranking errors in queries 1–4, bringing the most information-dense paragraphs to the top.
  - **Known Limitation (Figure Query):** For query 5 ("FIGURE 3 AI RMF NIST"), the Cross-Encoder assigned higher relevance to surrounding narrative/explanatory paragraphs describing the figure than to the concise figure caption itself. This is expected behavior for a passage-trained cross-encoder (trained on web text pairs) operating over heterogeneous scientific document elements where captions are structurally short and information-dense. This behavior is treated as an **active known limitation**, not a solved problem. Future directions include caption-specific prompt templates or fine-tuning on scientific document pairs.
- **Changed Files:**
  - `PROJECT_PROGRESS.md` (Updated)
  - Scratch evaluation scripts (disposable — not part of application code)
- **Commands & Actual Test Results:**
  - No new tests added (evaluation task — documentation only).
  - All 406 project tests continue to pass.
- **Decisions Made:**
  - **No Architecture Changes:** Evaluation findings are recorded as documentation; no retrieval code, tests, or indices were modified.
  - **Limitation Preserved:** The figure/caption preference behavior is recorded honestly as a known limitation and future evaluation direction rather than worked around heuristically.
- **Remaining Limitations:**
  - Cross-Encoder can prefer explanatory narrative text over exact figure captions for figure-focused queries. Caption-aware reranking or domain fine-tuning is a future direction.
  - Vector index persistence (save/load) is not yet implemented.
  - LLM generation integration and UI remain part of later work.
- **Blockers:**
  - None.

---

## Handoff

- **Status:** Day 16 is complete (Tasks 1, 2, and 3 all complete and verified).
- **Next Planned Day (Day 17):** Continued reranking improvements or evaluation framework (per roadmap: Days 18–22 cover Ragas evaluation, ground truth construction, and retrieval/generation metrics).


