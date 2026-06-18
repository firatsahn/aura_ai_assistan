# RAG API image (Step 3a). Build context is the project root so it can pull in
# both requirements.txt and the backend/ package:
#   docker compose up --build backend
FROM python:3.12-slim

WORKDIR /app

# Install deps first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (the importable `backend` package).
COPY backend/ ./backend/

# Default API port; overridable but the compose file maps 4242:4242.
EXPOSE 4242

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "4242"]
