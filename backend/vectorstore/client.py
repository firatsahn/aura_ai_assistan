"""Qdrant client wrapper: collection schema, upsert, and dense search.

The collection stores one point per chunk. Each point's payload carries the
chunk's full metadata (source_doc, doc_type, modality, page, section, the text
itself, ...) because retrieval needs it twice: to build citations in the answer
and to filter results (e.g. by doc_type or source). Similarity is cosine, the
usual choice for normalized text embeddings.

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

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "aura_corpus")

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


def ensure_collection(
    client: QdrantClient,
    name: str = COLLECTION,
    size: int = EMBEDDING_DIM,
    distance: qm.Distance = qm.Distance.COSINE,
) -> None:
    """Create the collection if it does not already exist (idempotent)."""
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=size, distance=distance),
    )


def upsert(
    client: QdrantClient,
    chunks: list[Chunk],
    vectors: list[list[float]],
    name: str = COLLECTION,
) -> int:
    """Upsert one point per chunk; payload carries the full chunk metadata."""
    if len(chunks) != len(vectors):
        raise ValueError(f"chunk/vector count mismatch: {len(chunks)} != {len(vectors)}")

    points = [
        qm.PointStruct(
            id=_point_id(chunk.chunk_id),
            vector=vector,
            payload=chunk.to_dict(),
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=name, points=points)
    return len(points)


def search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 5,
    query_filter: qm.Filter | None = None,
    name: str = COLLECTION,
) -> list[SearchHit]:
    """Dense (cosine) nearest-neighbor search. Returns hits with payloads."""
    results = client.query_points(
        collection_name=name,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    ).points

    hits: list[SearchHit] = []
    for r in results:
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
