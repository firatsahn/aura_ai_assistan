"""OpenAI embedding client: text -> vectors, batched.

One public function, `embed(texts)`, used by both indexing (backend/index.py)
and query time. Keeping it in a single module means the provider can change in
one place without touching the rest of the system, and guarantees the same
model embeds documents and queries (a hard requirement — mismatched models
yield vectors that don't share a space, breaking retrieval).

The default model is OpenAI `text-embedding-3-small` (1536 dims). The corpus is
Turkish; this model's multilingual coverage is adequate for the baseline. The
provider is selected by env (`EMBEDDING_MODEL`, `OPENAI_API_KEY`), mirroring the
ingestion vision module.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # load credentials / config from project .env if present

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# Native output dimensionality per model. The collection schema reads this so it
# can never drift from the client.
_MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
EMBEDDING_DIM = _MODEL_DIMS.get(EMBEDDING_MODEL, 1536)

# Inputs per API call. Hundreds of chunks come through here; one-by-one calls
# are slow and costly. The endpoint accepts far more, but 100 keeps requests
# small and progress legible.
_BATCH_SIZE = 100


def embed(texts: list[str]) -> list[list[float]]:
    """Embed `texts` with the configured model, in batches, preserving order."""
    if not texts:
        return []

    client = _client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        # The API guarantees results come back in the input order, but sort by
        # index defensively so a future provider swap can't silently misalign.
        for item in sorted(response.data, key=lambda d: d.index):
            vectors.append(item.embedding)
    return vectors


def _client():
    import openai

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set — needed for embeddings "
            f"(model={EMBEDDING_MODEL}). Add it to .env."
        )
    return openai.OpenAI()
