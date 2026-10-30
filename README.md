# PaperMind

Multimodal scientific research paper assistant that answers questions about academic papers using grounded retrieval and generation.

## Docker Setup

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with Docker Compose (included in Docker Desktop).

### 1. Clone the repository

```bash
git clone <repo-url>
cd RESEARCH_PROJECT
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and add your Groq API key:

```text
GROQ_API_KEY=gsk_your_key_here
```

> [!CAUTION]
> Never commit `.env` or any file containing secrets. `.env` is already listed in `.gitignore`.

### 3. Add a sample PDF

Place the PDF you want to query at:

```
data/sample_paper.pdf
```

The `data/` directory is bind-mounted into the container as read-only. PDF files are excluded from Git (see `.gitignore`), so the file must be added manually.

### 4. Start the service

```bash
docker compose up --build
```

On first startup, Hugging Face embedding models are downloaded automatically. Subsequent starts are faster because the models are cached in the `hf_cache` named volume.

### 5. Verify the service is running

Check service status:

```bash
docker compose ps
```

Check the health endpoint:

```
http://localhost:8000/health
```

Expected response: `{"status": "healthy"}`

### 6. Query the API (PowerShell)

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/query" `
  -ContentType "application/json" `
  -Body '{"pdf_path": "sample_paper.pdf", "query": "What is the main contribution of this paper?", "top_k": 5}'
```

The `pdf_path` is resolved relative to `data/` inside the container (`PAPER_MIND_DOC_ROOT=/app/data`).

### 7. Stop the service

```bash
docker compose down
```

> [!WARNING]
> Do **not** run `docker compose down -v` if you want to preserve the Hugging Face model cache. The `-v` flag removes the `hf_cache` named volume, requiring a full model re-download on the next start.

### Persistence notes

| What | Where | Persisted? |
|---|---|---|
| Hugging Face model cache | `hf_cache` named Docker volume | Yes — survives `docker compose down` |
| PDFs | `./data` bind mount (read-only) | Yes — your local `data/` directory |
| FAISS indexes | In-memory, rebuilt per request | No — rebuilt on each query |

