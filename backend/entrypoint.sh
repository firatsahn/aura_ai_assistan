#!/bin/sh
# Container start: guarded auto-index, then serve the API + frontend.
#
# Step 2 (embed + upsert) runs only when the Qdrant collection is empty, so a
# fresh volume gets indexed once and restarts reuse it (idempotent, but skipping
# avoids needless OpenAI embedding cost). Step 1 (ingestion/vision) is NOT run
# here — chunks.jsonl is prebuilt and copied into the image.
set -e

echo "[entrypoint] checking Qdrant collection ..."
COUNT=$(python - <<'PY'
from backend.vectorstore import get_client, COLLECTION
try:
    c = get_client()
    n = c.count(COLLECTION).count if c.collection_exists(COLLECTION) else 0
except Exception:
    n = 0
print(n)
PY
)

if [ "$COUNT" -gt 0 ]; then
    echo "[entrypoint] collection already has $COUNT points — skipping indexing."
else
    echo "[entrypoint] empty collection — embedding + upserting chunks ..."
    python -m backend.index --in data/chunks.jsonl
fi

echo "[entrypoint] starting API on :4242"
exec uvicorn backend.api:app --host 0.0.0.0 --port 4242
