"""Pipeline retrieval modes + abstention gate. Hermetic (no Qdrant / network).

Monkeypatches the four collaborators answer_question wires together so the test
exercises only the orchestration: which mode reaches search(), and that the
abstain/answer gate reads the dense cosine score even in hybrid mode.
"""

import backend.pipeline as pipeline
from backend.sparse import SparseVector
from backend.vectorstore.client import SearchHit


def _hit(score: float, cid: str = "a::000") -> SearchHit:
    return SearchHit(score=score, chunk_id=cid, text="t", source_doc="a.md", payload={})


def _patch(monkeypatch, *, search, dense_gate=None, generate=lambda q, h: "ANSWER"):
    """Wire stubs; `search` is called with the real kwargs so we can assert them."""
    calls = []

    def fake_search(client, qvec, *, query_sparse=None, retrieval_mode="dense", top_k=5):
        calls.append({"mode": retrieval_mode, "sparse": query_sparse, "top_k": top_k})
        if retrieval_mode == "dense" and top_k == 1 and dense_gate is not None:
            return dense_gate
        return search

    monkeypatch.setattr(pipeline, "get_client", lambda: object())
    monkeypatch.setattr(pipeline, "embed", lambda texts: [[0.0, 0.0]])
    monkeypatch.setattr(pipeline, "encode_query", lambda t: SparseVector([1], [1.0]))
    monkeypatch.setattr(pipeline, "search", fake_search)
    monkeypatch.setattr(pipeline, "generate", generate)
    return calls


def test_dense_mode_threads_through(monkeypatch):
    calls = _patch(monkeypatch, search=[_hit(0.50)])
    out = pipeline.answer_question("soru", retrieval_mode="dense")
    assert out["abstained"] is False
    assert [c["mode"] for c in calls] == ["dense"]
    assert calls[0]["sparse"] is None


def test_hybrid_mode_passes_sparse_and_uses_dense_gate(monkeypatch):
    # Hybrid hits carry a tiny RRF score (0.02) — below the 0.38 cosine threshold.
    # The separate dense gate scores 0.50, so the call must NOT abstain: proof the
    # gate reads dense cosine, not the fused RRF score.
    calls = _patch(monkeypatch, search=[_hit(0.02, "b::000")], dense_gate=[_hit(0.50)])
    out = pipeline.answer_question("E-102 nedir", retrieval_mode="hybrid")

    assert out["abstained"] is False
    assert out["answer"] == "ANSWER"
    modes = [c["mode"] for c in calls]
    assert "hybrid" in modes and "dense" in modes          # ranking + gate probes
    hybrid_call = next(c for c in calls if c["mode"] == "hybrid")
    assert hybrid_call["sparse"] is not None                # sparse threaded through


def test_abstains_when_dense_gate_below_threshold(monkeypatch):
    generated = []
    calls = _patch(
        monkeypatch,
        search=[_hit(0.02, "b::000")],
        dense_gate=[_hit(0.30)],                            # below 0.38
        generate=lambda q, h: generated.append(1) or "SHOULD-NOT-RUN",
    )
    out = pipeline.answer_question("alakasiz soru", retrieval_mode="hybrid")

    assert out["abstained"] is True
    assert out["answer"] == pipeline.ABSTENTION_MESSAGE
    assert out["sources"] == []
    assert generated == []                                  # generate not called
