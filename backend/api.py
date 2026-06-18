"""HTTP API for the RAG system — the single entry point for frontend and eval.

One real route, `POST /query`: a question in, a grounded answer (or an
abstention) plus its sources out. The work lives in `backend.pipeline`; this
module is only the HTTP shell so the answer path stays testable without a
server. `GET /health` exists for the Docker healthcheck and smoke tests.

    uvicorn backend.api:app --port 8000
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.pipeline import answer_question

app = FastAPI(title="Aura RAG API", version="0.1.0")

# Open CORS for now so a local test page / the future frontend (:3000) can call
# the API from the browser. Tighten to specific origins before production.
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest) -> dict[str, Any]:
    return answer_question(
        req.question, top_k=req.top_k, retrieval_mode=req.retrieval_mode
    )
