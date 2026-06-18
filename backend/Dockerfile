# RAG API image (Step 3a + Step 5 frontend). Build context is the project root
# so it can pull in requirements.txt, the backend/ package, the prebuilt chunk
# list, the eval results, and the static frontend:
#   docker compose up --build backend
FROM python:3.12-slim

WORKDIR /app

# Install deps first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (the importable `backend` package).
COPY backend/ ./backend/

# Prebuilt artifacts the API serves: chunks for indexing (Step 2), eval results
# for /metrics, and the static single-page frontend served at /.
COPY data/chunks.jsonl ./data/chunks.jsonl
COPY eval/results.json ./eval/results.json
COPY frontend/ ./frontend/

# Startup script: guarded auto-index, then uvicorn.
RUN chmod +x ./backend/entrypoint.sh

# Default API port; overridable but the compose file maps 4242:4242.
EXPOSE 4242

ENTRYPOINT ["./backend/entrypoint.sh"]
