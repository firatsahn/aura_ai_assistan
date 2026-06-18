"""Vector store — thin Qdrant wrapper (collection schema, upsert, search).

The rest of the system talks to Qdrant only through this module, so the store
can be swapped or reconfigured in one place.
"""

from .client import (
    COLLECTION,
    SearchHit,
    ensure_collection,
    get_client,
    search,
    upsert,
)

__all__ = [
    "COLLECTION",
    "SearchHit",
    "ensure_collection",
    "get_client",
    "search",
    "upsert",
]
