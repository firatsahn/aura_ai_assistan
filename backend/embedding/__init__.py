"""Embedding provider — the single point that turns text into vectors.

Indexing and query-time both call `embed()` so the same model embeds both
sides; using different models would put the vectors in different spaces and
make search meaningless.
"""

from .client import EMBEDDING_DIM, EMBEDDING_MODEL, embed

__all__ = ["embed", "EMBEDDING_MODEL", "EMBEDDING_DIM"]
