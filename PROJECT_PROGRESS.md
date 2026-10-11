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
| **Embeddings** | Sentence Transformers | Planned (Unverified) |
| **Vector Storage** | FAISS (initially) | Planned (Unverified) |
| **LLM Inference** | LLM Provider or Local Alternative | Planned (Unverified) |
| **Evaluation** | Ragas | Planned (Unverified) |
| **Observability** | Langfuse | Planned (Unverified) |

---

## Current State

- **Status:** Day 3 Complete
- **Date:** October 11, 2026
- **Environment:** Windows, Python 3.13.5, active `.venv`
- **Version Control:** Git initialized locally; no commits made (files uncommitted/staged as requested)
- **Configuration & Dependencies:**
  - `requirements.txt`: Pinned to `pymupdf==1.28.2` and `pytest==9.1.1`
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

---

## Handoff

- **Status:** Day 3 is complete.
- **Next Session Instructions (Day 4):**
  1. Read this progress file (`PROJECT_PROGRESS.md`).
  2. Plan and implement embedding generation pipeline using Sentence Transformers (`sentence-transformers`).
  3. Define embedding model interface, batch encoding, and vector representations for `TextChunk` instances.
  4. Prepare embedding outputs for downstream vector indexing (FAISS).
