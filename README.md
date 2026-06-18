# Aura AI Assistant — Grounded RAG Support Assistant

> ⚠️ **Draft.** This README will be updated as the  progresses.

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

> _Diagram is a draft; final components are justified in `DECISIONS.md`._

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
| `ANTHROPIC_API_KEY` | Claude key — generation (`claude-opus-4-8`) and Claude vision | Yes      |
| `OPENAI_API_KEY`    | OpenAI key — embeddings (`text-embedding-3-small`)            | Yes      |
| `EMBEDDING_MODEL`   | Embedding model override (default `text-embedding-3-small`)   | No       |
| `QDRANT_URL`        | Vector store URL (`http://localhost:6333`, or `http://qdrant:6333` in Compose) | No |
| `QDRANT_COLLECTION` | Qdrant collection name (default `aura_corpus`)               | No       |
| `GEMINI_API_KEY`    | Gemini key — alternative vision provider (`VISION_PROVIDER=gemini`) | No  |

> _Full list will be finalized as the implementation progresses._

## Setup & Run (Docker)

The whole system (backend + frontend + vector store) comes up with a single command:

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## Ingestion (Indexing the Corpus)

Processes the corpus (PDF, images, Excel/CSV, markdown) and loads it into the vector
store, preserving the structure and context of visual and tabular content.

```bash
# 1. Start the vector store (Qdrant). Web UI: http://localhost:6333/dashboard
docker compose up -d qdrant

# 2. Step 1 — read the corpus into a tagged chunk list
python -m backend.ingestion.run --doc-dir doc --out data/chunks.jsonl

# 3. Step 2 — embed every chunk and load it into Qdrant
python -m backend.index

# Verify: a query whose answer lives only in an image (the LED card, doc 03)
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
├── backend/          # RAG API (ingestion, retrieval, generation)
├── frontend/         # Simple web UI (question → answer → sources)
├── eval/             # Evaluation harness + golden set
├── docker-compose.yml
├── README.md
├── DECISIONS.md      # Design & rationale document
├── brief.pdf         # Case study brief (git-ignored)
└── doc/              # Corpus documents (git-ignored)
```

> _Structure is a draft and will be updated during development._

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
