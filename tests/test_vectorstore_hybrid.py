"""Hybrid vector store: schema, dense parity, the lexical win. Hermetic.

Runs against an in-memory Qdrant (`QdrantClient(":memory:")`) — no server, no
network. If a given build lacks sparse / fusion / IDF support, the tests skip
cleanly rather than fail, mirroring tests/test_corpus.py's skip-on-absence style.
"""

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from backend.ingestion.schema import Chunk
from backend.sparse import encode_documents, encode_query
from backend.vectorstore import client as vc

NAME = "test_corpus"
SIZE = 4  # tiny dense vectors keep the fixtures readable


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, text=text, source_doc=f"{cid}.md",
                 doc_type="text", modality="markdown")


@pytest.fixture()
def store():
    client = QdrantClient(":memory:")
    try:
        vc.ensure_collection(client, name=NAME, size=SIZE)
    except Exception as exc:  # local build without sparse/IDF support
        pytest.skip(f"in-memory Qdrant lacks hybrid support: {exc}")
    return client


def test_schema_has_dense_and_sparse(store):
    cfg = store.get_collection(NAME).config.params
    assert vc.DENSE in cfg.vectors
    assert vc.SPARSE in (cfg.sparse_vectors or {})


def test_ensure_collection_rebuilds_dense_only_schema():
    client = QdrantClient(":memory:")
    # Old baseline schema: a single unnamed dense vector, no sparse.
    client.create_collection(
        collection_name=NAME,
        vectors_config=qm.VectorParams(size=SIZE, distance=qm.Distance.COSINE),
    )
    assert not vc._has_sparse(client, NAME)
    vc.ensure_collection(client, name=NAME, size=SIZE)  # detects + rebuilds
    assert vc._has_sparse(client, NAME)


def test_dense_search_returns_cosine_nearest(store):
    chunks = [_chunk("a::000", "alfa"), _chunk("b::000", "beta")]
    dense = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    vc.upsert(store, chunks, dense, encode_documents([c.text for c in chunks]), name=NAME)

    hits = vc.search(store, [1.0, 0.0, 0.0, 0.0], retrieval_mode="dense", top_k=1, name=NAME)
    assert [h.chunk_id for h in hits] == ["a::000"]


def test_hybrid_surfaces_exact_token_miss(store):
    # A: dense-near the query, no exact token. C: dense-mid, no exact token.
    # B: dense-far, but its text holds the literal "E-102" the query asks for.
    chunks = [
        _chunk("a::000", "internet baglantisi genel aciklama"),
        _chunk("c::000", "wifi ag ayarlari aciklama"),
        _chunk("b::000", "E-102 hata kodu aciklamasi"),
    ]
    dense = [
        [1.0, 0.0, 0.0, 0.0],   # a — nearest
        [0.8, 0.2, 0.0, 0.0],   # c — second
        [0.0, 0.0, 0.0, 1.0],   # b — far
    ]
    vc.upsert(store, chunks, dense, encode_documents([c.text for c in chunks]), name=NAME)

    qvec = [1.0, 0.0, 0.0, 0.0]
    dense_hits = vc.search(store, qvec, retrieval_mode="dense", top_k=2, name=NAME)
    hybrid_hits = vc.search(store, qvec, query_sparse=encode_query("E-102 nedir"),
                            retrieval_mode="hybrid", top_k=2, name=NAME)

    dense_ids = {h.chunk_id for h in dense_hits}
    hybrid_ids = {h.chunk_id for h in hybrid_hits}
    assert "b::000" not in dense_ids        # dense alone misses the exact-token chunk
    assert "b::000" in hybrid_ids           # hybrid surfaces it via lexical match


def test_hybrid_requires_query_sparse(store):
    with pytest.raises(ValueError):
        vc.search(store, [1.0, 0.0, 0.0, 0.0], retrieval_mode="hybrid", name=NAME)


def test_unknown_mode_raises(store):
    with pytest.raises(ValueError):
        vc.search(store, [1.0, 0.0, 0.0, 0.0], retrieval_mode="bogus", name=NAME)


def test_upsert_length_mismatch_raises(store):
    chunks = [_chunk("a::000", "alfa")]
    with pytest.raises(ValueError):
        vc.upsert(store, chunks, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                  encode_documents(["alfa"]), name=NAME)
