"""Retrieval metrics — Recall@k and MRR, dense vs hybrid, with no LLM calls.

This is the cheap, deterministic half of the harness and the evidence for the
Step 3b hybrid decision. It deliberately bypasses `answer_question()` (which
would trigger generation) and drives retrieval directly through `embed()` +
`search()`, the same two calls the pipeline makes before generation. Dense mode
queries the dense vector only; hybrid mode adds the BM25 sparse query and fuses
with RRF — identical to production retrieval.

A hit is "any expected source present": a ranked result counts if at least one
of its `source_doc`s is in `expected_sources`. Multi-hop questions list several
sources; we credit the retriever for surfacing any of them (intersection
non-empty), matching the roadmap's rule.

The scoring math (`recall_at_k`, `reciprocal_rank`) is pure and takes a ranked
list of source filenames, so it is unit-tested without Qdrant or the network.
"""

from __future__ import annotations

from typing import Any

from backend.embedding import embed
from backend.sparse import encode_query
from backend.vectorstore import get_client, search


def recall_at_k(ranked_sources: list[str], expected: list[str], k: int) -> float:
    """1.0 if any expected source appears in the first `k` results, else 0.0."""
    return 1.0 if set(ranked_sources[:k]) & set(expected) else 0.0


def reciprocal_rank(ranked_sources: list[str], expected: list[str]) -> float:
    """1 / (rank of the first expected source), or 0.0 if none is retrieved."""
    expected_set = set(expected)
    for i, src in enumerate(ranked_sources, 1):
        if src in expected_set:
            return 1.0 / i
    return 0.0


def retrieve_sources(
    client: Any, question: str, mode: str, top_k: int
) -> list[str]:
    """Run retrieval for one question, returning the ranked `source_doc` list."""
    [query_vector] = embed([question])
    if mode == "hybrid":
        hits = search(
            client,
            query_vector,
            query_sparse=encode_query(question),
            retrieval_mode="hybrid",
            top_k=top_k,
        )
    else:
        hits = search(client, query_vector, retrieval_mode="dense", top_k=top_k)
    return [h.source_doc for h in hits]


def evaluate_retrieval(
    answerable: list[dict[str, Any]],
    mode: str,
    ks: tuple[int, ...] = (3, 5),
) -> dict[str, Any]:
    """Recall@k (for each k) and MRR over the answerable questions in `mode`.

    `top_k` for retrieval is `max(ks)` so every requested cutoff is covered by a
    single search per question.
    """
    client = get_client()
    top_k = max(ks)

    per_question: list[dict[str, Any]] = []
    recall_sums = {k: 0.0 for k in ks}
    rr_sum = 0.0

    for q in answerable:
        ranked = retrieve_sources(client, q["question"], mode, top_k)
        expected = q["expected_sources"]
        recalls = {k: recall_at_k(ranked, expected, k) for k in ks}
        rr = reciprocal_rank(ranked, expected)
        for k in ks:
            recall_sums[k] += recalls[k]
        rr_sum += rr
        per_question.append(
            {
                "id": q["id"],
                "expected_sources": expected,
                "retrieved": ranked,
                "recall": recalls,
                "reciprocal_rank": rr,
            }
        )

    n = len(answerable) or 1
    return {
        "mode": mode,
        "n": len(answerable),
        "recall_at_k": {k: recall_sums[k] / n for k in ks},
        "mrr": rr_sum / n,
        "per_question": per_question,
    }
