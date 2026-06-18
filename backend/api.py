"""HTTP API for the RAG system — the single entry point for frontend and eval.

The answer path is `POST /query`: a question in, a grounded answer (or an
abstention) plus its sources out (and, with `debug`, a step-by-step trace). The
work lives in `backend.pipeline`; this module is only the HTTP shell so the
answer path stays testable without a server.

Two read-only helpers back the frontend tabs: `GET /metrics` serves the eval
results (dense vs hybrid), and the static frontend is mounted at `/`. `GET
/health` exists for the Docker healthcheck and smoke tests.

    uvicorn backend.api:app --port 4242
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.pipeline import answer_question

# Repo layout: this file is backend/api.py, so the project root is two levels up.
_ROOT = Path(__file__).resolve().parent.parent
# Overridable so the Docker image can point at its own copied paths.
METRICS_PATH = Path(os.environ.get("METRICS_PATH", _ROOT / "eval" / "results.json"))
FRONTEND_DIR = Path(os.environ.get("FRONTEND_DIR", _ROOT / "frontend"))

app = FastAPI(title="Aura RAG API", version="0.1.0")

# Open CORS for now so a local test page / the future frontend can call the API
# from the browser. Tighten to specific origins before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Kullanıcının sorusu")
    top_k: int = Field(5, ge=1, le=20, description="Getirilecek chunk sayısı")
    retrieval_mode: Literal["dense", "hybrid"] = Field(
        "dense", description="Retrieval: dense baseline veya hybrid (dense+sparse)"
    )
    debug: bool = Field(
        False, description="True ise ara adımları (trace) cevaba ekler"
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest) -> dict[str, Any]:
    return answer_question(
        req.question,
        top_k=req.top_k,
        retrieval_mode=req.retrieval_mode,
        debug=req.debug,
    )


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Serve the latest eval results (dense vs hybrid) for the Metrics tab."""
    if not METRICS_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Henüz eval sonucu yok ({METRICS_PATH.name}). "
            "Önce `python -m eval.run` çalıştırın.",
        )
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


# Mount the static frontend LAST so explicit API routes above take precedence;
# "/" then serves index.html (html=True) and the rest of the assets.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
