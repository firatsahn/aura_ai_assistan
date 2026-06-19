# Aura AI Assistant — Grounded RAG Support Assistant

**🇬🇧 English** · [🇹🇷 Türkçe](README.tr.md)

A Retrieval-Augmented Generation (RAG) support assistant for a B2B SaaS scenario.
It answers end-user questions strictly from a customer's knowledge base and **every
answer is grounded in the corpus and cites its sources**. When the system cannot answer
reliably from the retrieved context, it **abstains** instead of guessing.

The corpus for this case study is built around the **Aura Hub G2** smart-home device:
setup guides, subscription/return policy, LED status indicators, error codes, technical
specifications, a detailed user manual, a troubleshooting FAQ, and a privacy & data
policy. The corpus is intentionally heterogeneous — plain text and structured documents,
multi-page PDFs, scanned pages, screenshots, images with embedded text/tables, and
spreadsheet files (Excel/CSV). **Some information exists only inside images or tables and
is not repeated in any plain-text document**, so visual and tabular content must be
extracted with its structure and context preserved.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Corpus](#corpus)
- [Requirements](#requirements)
- [Environment Variables](#environment-variables)
- [Setup & Run (Docker)](#setup--run-docker)
- [Ingestion (Indexing the Corpus)](#ingestion-indexing-the-corpus)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Key Decisions](#key-decisions)
- [Evaluation Results](#evaluation-results)

---

## Architecture Overview

```
User ──> Frontend ──> Backend API
                          │
            ┌─────────────┼──────────────┐
            │             │              │
        Retrieval    Generation      Abstention
       (hybrid +     (LLM, answer    (low confidence
        reranking)    with citations) → no answer)
            │
      Vector Store  <── Ingestion ── Corpus (PDF / image / Excel / md)
```

> Component choices are justified in [`DECISIONS.md`](DECISIONS.md).

## Corpus

The knowledge base (kept locally, **not** committed — see [.gitignore](.gitignore)):

| File                               | Type            | Notes                                  |
|------------------------------------|-----------------|----------------------------------------|
| `01_aura_kurulum_kilavuzu.md`      | Markdown        | Setup & getting-started guide          |
| `02_abonelik_ve_iade_politikasi`   | PDF             | Subscription & return policy           |
| `03_led_durum_gostergeleri`        | PNG (image)     | LED status indicators (visual only)    |
| `04_hata_kodlari`                  | XLSX            | Error codes (tabular)                  |
| `05_teknik_ozellikler_scan`        | PDF (scanned)   | Technical specs (scanned page)         |
| `06_detayli_kullanici_kilavuzu`    | PDF             | Detailed user manual                   |
| `07_sorun_giderme_sss`             | PDF             | Troubleshooting FAQ                    |
| `08_gizlilik_ve_veri_politikasi`   | PDF             | Privacy & data policy                  |


## Requirements

- Docker and Docker Compose
- _(For local development)_ Python 3.11+ and Node.js 20+

## Environment Variables

Copy `.env.example` to `.env` and fill it in:

```bash
cp .env.example .env
```

| Variable            | Description                                                  | Required |
|---------------------|--------------------------------------------------------------|----------|
| `OPENAI_API_KEY`    | OpenAI key — embeddings (`text-embedding-3-small`) **and** generation (`gpt-4o-mini`) | Yes |
| `ANTHROPIC_API_KEY` | Claude key — default vision provider for scanned/image docs (`claude-opus-4-8`) | Yes¹ |
| `GEMINI_API_KEY`    | Gemini key — alternative vision provider (`VISION_PROVIDER=gemini`) | Yes¹ |
| `VISION_PROVIDER`   | `anthropic` (default) or `gemini`; blank auto-detects from whichever key is set | No |
| `GENERATION_MODEL`  | Generation model override (default `gpt-4o-mini`)            | No       |
| `EMBEDDING_MODEL`   | Embedding model override (default `text-embedding-3-small`)   | No       |
| `ABSTENTION_THRESHOLD` | Below this top retrieval score the system abstains (default `0.38`) | No |
| `QDRANT_URL`        | Vector store URL (`http://localhost:6333`, or `http://qdrant:6333` in Compose) | No |
| `QDRANT_COLLECTION` | Qdrant collection name (default `aura_corpus`)               | No       |

> ¹ A vision key is only needed to **re-run ingestion** on the raw corpus. The
> prebuilt `data/chunks.jsonl` is committed, so a normal `docker compose up` boot
> indexes from it and needs **only `OPENAI_API_KEY`** (embeddings + generation).

## Setup & Run (Docker)

The entire system — backend, frontend, and vector store — is brought up with a single
command. `docker-compose.yml` defines two services: **Qdrant** (vector store) and
**backend** (the RAG API, which also serves the static frontend; no separate frontend
container is required).

### 1. Initial build

```bash
cp .env.example .env      # then set OPENAI_API_KEY (see Environment Variables)
docker compose up --build
```

The `--build` flag compiles the backend image. On startup, the backend performs a
**guarded auto-index** (`backend/entrypoint.sh`): if the Qdrant collection is empty, it
embeds the prebuilt `data/chunks.jsonl`, upserts the vectors, and then serves the API.
The costly ingestion/vision pipeline is **not** re-run — the chunk list is committed and
baked into the image.

Once the stack is up, the following endpoints are available:

- **Application (UI + API):** http://localhost:4242
- **API endpoints:** `POST /query`, `GET /metrics`, `GET /decisions/{tr|en}`, `GET /health`, `GET /docs`
- **Qdrant dashboard:** http://localhost:6333/dashboard

### 2. Subsequent runs

After the images are built, no rebuild is necessary. Qdrant data is persisted in a named
Docker volume (`qdrant_storage`), so the index is reused across restarts and the
auto-index step is skipped automatically when a populated collection is detected:

```bash
docker compose up            # start (reuses built images and indexed volume)
docker compose up -d         # start in detached (background) mode
docker compose down          # stop (volume retained → data persists)
docker compose down -v       # stop and wipe the Qdrant volume (forces a re-index on next boot)
```

Rebuild only after code changes: `docker compose up --build`.

### 3. Running from a fresh git clone (with a `.env` you were given)

This is the common handover case: someone shares a ready `.env` with you, you clone the
repository, and you want it running. The project is self-contained — the corpus index
(`data/chunks.jsonl`) is committed and baked into the image — so a clean clone plus a
valid `.env` is all that is required. No ingestion or vision step is involved.

```bash
# 1. Clone the repository
git clone <repo> && cd project

# 2. Place the .env you were given at the project root (next to docker-compose.yml).
#    It is gitignored, so it never arrives via the clone — copy it in yourself:
cp /path/to/the/.env .env
#    ... or copy it from another machine:
#    scp user@host:/path/to/.env .env
#    (Alternatively, recreate it from the template: cp .env.example .env, then fill it in.)

# 3. Build and start
docker compose up --build
```

The app is then served at **http://localhost:4242**.

**Notes:**

- `.env` is **gitignored** and is never part of the clone — it must always be transferred
  out of band (shared file, `scp`, secrets manager) or recreated from `.env.example`.
- Only `OPENAI_API_KEY` is required for a normal boot (embeddings + generation). A vision
  key (`ANTHROPIC_API_KEY` or `GEMINI_API_KEY`) is needed solely to re-run ingestion on
  the raw `doc/` corpus.
- Within Compose, the backend reaches Qdrant by service name; `QDRANT_URL` is overridden
  to `http://qdrant:6333` automatically, so it need not be set in `.env`.

## Ingestion (Indexing the Corpus)

This step processes the corpus (PDF, images, Excel/CSV, markdown) and loads it into the
vector store, preserving the structure and context of visual and tabular content. It is
only required when re-indexing the raw corpus; a standard boot uses the prebuilt
`data/chunks.jsonl`.

```bash
# 1. Start the vector store (Qdrant). Dashboard: http://localhost:6333/dashboard
docker compose up -d qdrant

# 2. Read the corpus into a tagged chunk list
python -m backend.ingestion.run --doc-dir doc --out data/chunks.jsonl

# 3. Embed each chunk and load it into Qdrant
python -m backend.index

# Verify with a query whose answer exists only in an image (the LED card, doc 03)
python -m backend.index --query "internet bağlantısı yok"
```

## Evaluation

A golden set of 45 question/answer pairs (`eval/golden_set.jsonl`) measures retrieval
quality (recall@k, MRR), generation quality (faithfulness/groundedness, answer
relevance) and abstention accuracy — **dense vs hybrid side by side** — with a single
command and no external eval framework:

```bash
python -m eval.run                 # full: retrieval + generation (LLM judge) + abstention
python -m eval.run --retrieval-only  # free, deterministic retrieval metrics only (no LLM)
python -m eval.run --no-judge        # answers + abstention, skip the LLM judge
```

Results, run config and per-question detail are written to `eval/results.json`. The
LLM judge runs on `gpt-4o` (stronger than the `gpt-4o-mini` generator, to limit
self-bias). See **[`DECISIONS.md`](DECISIONS.md#evaluation-step-4)** for methodology.

## Project Structure

```
.
├── backend/          # RAG API (ingestion, retrieval, generation) + serves the frontend
├── frontend/         # Static web UI (chat, metrics, architecture, prompt-flow, decisions)
├── eval/             # Evaluation harness + golden set
├── docker-compose.yml
├── README.md         # This file (English)
├── README.tr.md      # Turkish README
├── DECISIONS.md      # Design & rationale document (English)
├── DECISIONS.tr.md   # Design & rationale document (Turkish)
├── brief.pdf         # Case study brief (git-ignored)
└── doc/              # Corpus documents (git-ignored)
```

## Key Decisions

Rationale for chunking, embedding model, vector store, retrieval strategy, visual/table
handling, and the production migration plan is documented in
**[`DECISIONS.md`](DECISIONS.md)**.

## Evaluation Results

Measured on the 45-question golden set. Hybrid retrieval wins on every retrieval
metric and on generation quality — the measured justification for the Step 3b decision.

| Metric             | Dense (baseline) | Hybrid   |
|--------------------|------------------|----------|
| Recall@3           | 0.93             | **0.97** |
| Recall@5           | 0.95             | **0.97** |
| MRR                | 0.81             | **0.90** |
| Faithfulness       | 0.96             | **0.99** |
| Answer Relevance   | 0.91             | **0.92** |
| Abstention recall  | 5/5              | 5/5      |
| False abstentions  | 3/40             | 3/40     |

> Reproduce with `python -m eval.run`. Judge model: `gpt-4o`; generation: `gpt-4o-mini`.
