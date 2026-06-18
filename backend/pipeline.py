"""RAG orchestration: question -> embed -> search -> abstain or generate.

The single answer path, kept out of the HTTP layer so it can be exercised
directly (and later by the eval harness) without standing up a server. It wires
together the three existing pieces — `embed()` (embedding), `search()`
(vectorstore), `generate()` (generation) — and adds the one piece Step 3a owns:
abstention.

Abstention is a single retrieval-score rule, not a second LLM call. If the best
hit's similarity is below `ABSTENTION_THRESHOLD`, the corpus has nothing
relevant, so we never call the model and answer with a fixed "couldn't find it"
message. This is what stops the system from hallucinating on out-of-corpus
questions, and saves a model call. The grounded prompt in `generate()` is a
second, independent safety net (it abstains too when context is insufficient).

The default 0.38 was calibrated on this corpus: relevant queries scored 0.40–0.64
while out-of-corpus queries scored 0.29–0.34, leaving a clean gap. It stays
env-tunable (`ABSTENTION_THRESHOLD`).
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from backend.embedding import EMBEDDING_MODEL, embed
from backend.generation import generate
from backend.generation.client import GENERATION_MODEL, _format_context
from backend.sparse import encode_query
from backend.vectorstore import SearchHit, get_client, search

load_dotenv()

ABSTENTION_THRESHOLD = float(os.environ.get("ABSTENTION_THRESHOLD", "0.38"))

# Fixed message so abstention is recognizable to callers (frontend / eval).
ABSTENTION_MESSAGE = "Bu bilgi tabanında bunu bulamadım."

# How much chunk text to keep in trace/source previews (chars).
_TRACE_SNIPPET = 240


def answer_question(
    question: str, top_k: int = 5, retrieval_mode: str = "dense", debug: bool = False
) -> dict[str, Any]:
    """Retrieve, then either abstain (low score) or generate a grounded answer.

    Abstention stays a dense cosine gate in both modes: the 0.38 threshold was
    calibrated on cosine scores, but hybrid hits carry RRF scores on a different
    scale. So hybrid changes the *ranking* handed to `generate()`, while the
    abstain/answer decision still rests on the dense top-1 similarity (one extra
    cheap dense probe locally).

    With `debug=True` the response carries a `trace` walking the pipeline step by
    step (embed -> search -> abstention -> generation) for the "Prompt akışı"
    tab. It does not change the answer path, only observes it.
    """
    client = get_client()
    [query_vector] = embed([question])

    trace: dict[str, Any] = {}
    if debug:
        trace["embedding"] = {
            "model": EMBEDDING_MODEL,
            "dim": len(query_vector),
            "preview": [round(v, 4) for v in query_vector[:8]],
        }

    if retrieval_mode == "hybrid":
        query_sparse = encode_query(question)
        hits = search(
            client,
            query_vector,
            query_sparse=query_sparse,
            retrieval_mode="hybrid",
            top_k=top_k,
        )
        gate = search(client, query_vector, retrieval_mode="dense", top_k=1)
        top_score = gate[0].score if gate else None
        if debug:
            trace["sparse"] = {"term_count": len(query_sparse.indices)}
    else:
        hits = search(client, query_vector, retrieval_mode="dense", top_k=top_k)
        top_score = hits[0].score if hits else None

    if debug:
        trace["retrieval"] = {
            "mode": retrieval_mode,
            "hits": [_trace_hit(rank, h) for rank, h in enumerate(hits, 1)],
        }

    abstained = not hits or top_score is None or top_score < ABSTENTION_THRESHOLD
    if debug:
        trace["abstention"] = {
            "threshold": ABSTENTION_THRESHOLD,
            "top_score": top_score,
            "decision": "abstain" if abstained else "answer",
        }

    if abstained:
        if debug:
            trace["generation"] = {"skipped": True}
        result: dict[str, Any] = {
            "answer": ABSTENTION_MESSAGE,
            "abstained": True,
            "sources": [],
            "top_score": top_score,
        }
        if debug:
            result["trace"] = trace
        return result

    answer = generate(question, hits)
    sources = [
        {
            "source_doc": h.source_doc,
            "section": h.payload.get("section"),
            "modality": h.payload.get("modality"),
            "score": h.score,
            "chunk_id": h.chunk_id,
            "key": h.payload.get("metadata", {}).get("key"),
            "text": h.text,
        }
        for h in hits
    ]
    if debug:
        trace["generation"] = {
            "skipped": False,
            "model": GENERATION_MODEL,
            "context": _format_context(hits),
            "answer": answer,
        }
    result = {
        "answer": answer,
        "abstained": False,
        "sources": sources,
        "top_score": top_score,
    }
    if debug:
        result["trace"] = trace
    return result


def _trace_hit(rank: int, h: SearchHit) -> dict[str, Any]:
    """A compact, JSON-friendly view of one hit for the trace."""
    text = h.text or ""
    return {
        "rank": rank,
        "score": h.score,
        "chunk_id": h.chunk_id,
        "source_doc": h.source_doc,
        "section": h.payload.get("section"),
        "modality": h.payload.get("modality"),
        "text": text[:_TRACE_SNIPPET] + ("…" if len(text) > _TRACE_SNIPPET else ""),
    }
