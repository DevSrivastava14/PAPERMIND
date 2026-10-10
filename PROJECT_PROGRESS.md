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

- **Status:** Day 1 Complete
- **Date:** October 10, 2026
- **Environment:** Windows, Python 3.13.5, active `.venv`
- **Version Control:** Git initialized locally; no commits made (files currently untracked)
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

---

## Handoff

- **Status:** Day 1 is complete.
- **Next Session Instructions:**
  1. Read this progress file (`PROJECT_PROGRESS.md`).
  2. Inspect the current repository state.
  3. Begin the existing roadmap's Day 2 task.
