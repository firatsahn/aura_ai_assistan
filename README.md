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

| Variable            | Description                              | Required |
|---------------------|------------------------------------------|----------|
| `LLM_API_KEY`       | API key for the generation model         | Yes      |
| `EMBEDDING_API_KEY` | API key for the embedding model (if any) | No       |
| `VECTOR_DB_URL`     | Vector store connection (if external)    | No       |

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
store, preserving the structure and context of visual and tabular content:

```bash
# TODO: ingestion command
```

## Evaluation

A golden set of ~30–50 question/answer pairs measures retrieval quality (recall@k, MRR)
and generation quality (faithfulness/groundedness, answer relevance), runnable with a
single command:

```bash
# TODO: evaluation command
```

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

| Metric            | Value |
|-------------------|-------|
| Recall@k          | _TBD_ |
| MRR               | _TBD_ |
| Faithfulness      | _TBD_ |
| Answer Relevance  | _TBD_ |

> _Results will be filled in once the evaluation harness is complete._
