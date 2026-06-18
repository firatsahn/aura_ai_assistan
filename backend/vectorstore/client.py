"""Qdrant client wrapper: collection schema, upsert, and hybrid search.

The collection stores one point per chunk with two vectors: a named *dense*
vector (cosine, semantic similarity) and a named *bm25* sparse vector (lexical,
exact-token matching). Retrieval runs either as dense-only (baseline) or as a
hybrid of the two fused with Reciprocal Rank Fusion (RRF) — see `search`.

Each point's payload carries the chunk's full metadata (source_doc, doc_type,
modality, page, section, the text itself, ...) because retrieval needs it twice:
to build citations in the answer and to filter results (e.g. by doc_type or
source).

Point IDs: a chunk_id like "04_hata_kodlari::002" is not a valid Qdrant point
ID (must be an unsigned int or a UUID), so we derive a deterministic
uuid5(namespace, chunk_id). Same chunk -> same point ID, so re-indexing
overwrites in place rather than duplicating (idempotent).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from backend.embedding import EMBEDDING_DIM
from backend.ingestion.schema import Chunk
from backend.sparse import SparseVector

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "aura_corpus")

# Named vectors on each point. Naming is required once a collection holds more
# than one vector, so the dense vector gets a name too.
DENSE = "dense"
SPARSE = "bm25"

# Candidate pool each arm pulls before fusion. Wider than top_k so a result
# ranked low by one arm but high by the other still reaches the fusion step.
_PREFETCH_LIMIT = 30

# Fixed namespace so chunk_id -> point ID is stable across runs and machines.
_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


@dataclass
class SearchHit:
    """One retrieved chunk with its similarity score and provenance payload."""

    score: float
    chunk_id: str
    text: str
    source_doc: str
    payload: dict[str, Any]


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def _has_sparse(client: QdrantClient, name: str) -> bool:
    """True if `name` already carries the sparse vector (i.e. new schema)."""
    sparse = client.get_collection(name).config.params.sparse_vectors or {}
    return SPARSE in sparse


def ensure_collection(
    client: QdrantClient,
    name: str = COLLECTION,
    size: int = EMBEDDING_DIM,
    distance: qm.Distance = qm.Distance.COSINE,
) -> None:
    """Create the collection with dense + sparse vectors (idempotent).

    Adding the sparse vector changed the schema, so a collection left over from
    the dense-only baseline is dropped and rebuilt — re-indexing repopulates it.
    """
    if client.collection_exists(name):
        if _has_sparse(client, name):
            return
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={DENSE: qm.VectorParams(size=size, distance=distance)},
        # IDF modifier: Qdrant weights each term by inverse document frequency
        # over the whole collection at query time, so the BM25 score is complete
        # without us storing or recomputing IDF.
        sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
    )


def upsert(
    client: QdrantClient,
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[SparseVector],
    name: str = COLLECTION,
) -> int:
    """Upsert one point per chunk with both vectors; payload carries metadata."""
    if not (len(chunks) == len(dense_vectors) == len(sparse_vectors)):
        raise ValueError(
            "chunk/dense/sparse count mismatch: "
            f"{len(chunks)} / {len(dense_vectors)} / {len(sparse_vectors)}"
        )

    points = [
        qm.PointStruct(
            id=_point_id(chunk.chunk_id),
            vector={
                DENSE: dense,
                SPARSE: qm.SparseVector(indices=sparse.indices, values=sparse.values),
            },
            payload=chunk.to_dict(),
        )
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors)
    ]
    client.upsert(collection_name=name, points=points)
    return len(points)


def _to_hits(points: list[Any]) -> list[SearchHit]:
    """Map Qdrant scored points to SearchHits, shared by both retrieval modes."""
    hits: list[SearchHit] = []
    for r in points:
        payload = r.payload or {}
        hits.append(
            SearchHit(
                score=r.score,
                chunk_id=payload.get("chunk_id", str(r.id)),
                text=payload.get("text", ""),
                source_doc=payload.get("source_doc", ""),
                payload=payload,
            )
        )
    return hits


def search(
    client: QdrantClient,
    query_vector: list[float],
    *,
    query_sparse: SparseVector | None = None,
    retrieval_mode: str = "dense",
    top_k: int = 5,
    query_filter: qm.Filter | None = None,
    name: str = COLLECTION,
) -> list[SearchHit]:
    """Retrieve hits, dense-only or hybrid (dense + sparse fused with RRF).

    In ``dense`` mode ``SearchHit.score`` is the cosine similarity (0–1). In
    ``hybrid`` mode it is the RRF fused score — a small rank-reciprocal value on
    a different scale, not comparable to cosine.
    """
    if retrieval_mode == "dense":
        results = client.query_points(
            collection_name=name,
            query=query_vector,
            using=DENSE,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
    elif retrieval_mode == "hybrid":
        if query_sparse is None:
            raise ValueError("hybrid retrieval requires query_sparse")
        results = client.query_points(
            collection_name=name,
            prefetch=[
                qm.Prefetch(query=query_vector, using=DENSE, limit=_PREFETCH_LIMIT),
                qm.Prefetch(
                    query=qm.SparseVector(
                        indices=query_sparse.indices, values=query_sparse.values
                    ),
                    using=SPARSE,
                    limit=_PREFETCH_LIMIT,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
    else:
        raise ValueError(f"unknown retrieval_mode: {retrieval_mode!r}")

    return _to_hits(results.points)
