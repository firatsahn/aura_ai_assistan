"""CLI: embed the chunk list and load it into Qdrant — the single index command.

    python -m backend.index                                   # read -> embed -> upsert
    python -m backend.index --query "internet bağlantısı yok"  # search (verify)

Combines Step 1's output (data/chunks.jsonl) with Step 2's embedding + vector
store. After it runs, the corpus is searchable. The same embed() backs both the
indexing path and --query, so queries land in the same vector space as the
documents.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.embedding import EMBEDDING_MODEL, embed
from backend.ingestion.schema import read_jsonl
from backend.sparse import encode_documents, encode_query
from backend.vectorstore import (
    COLLECTION,
    ensure_collection,
    get_client,
    search,
    upsert,
)


def build_index(chunks_path: str) -> int:
    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise SystemExit(f"no chunks in {chunks_path} — run the ingestion step first")

    print(f"Read {len(chunks)} chunks from {chunks_path}")
    print(f"Embedding with {EMBEDDING_MODEL} ...")
    texts = [c.text for c in chunks]
    vectors = embed(texts)
    sparse_vectors = encode_documents(texts)
    print(f"  embedded {len(vectors)} chunks (dim={len(vectors[0])}, + bm25 sparse)")

    client = get_client()
    ensure_collection(client)
    count = upsert(client, chunks, vectors, sparse_vectors)
    print(f"Upserted {count} points to collection '{COLLECTION}'")
    return count


def run_query(text: str, top_k: int, mode: str) -> None:
    client = get_client()
    [query_vector] = embed([text])
    query_sparse = encode_query(text) if mode == "hybrid" else None
    hits = search(
        client, query_vector, query_sparse=query_sparse, retrieval_mode=mode, top_k=top_k
    )

    print(f'Top {len(hits)} results for: "{text}" [{mode}]\n')
    for rank, hit in enumerate(hits, 1):
        section = hit.payload.get("section") or "-"
        snippet = " ".join(hit.text.split())[:160]
        print(f"{rank}. [{hit.score:.3f}] {hit.source_doc} — {section}")
        print(f"   {snippet}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="chunks", default="data/chunks.jsonl",
                        help="input chunk list (JSONL from the ingestion step)")
    parser.add_argument("--query", help="search the existing index instead of building it")
    parser.add_argument("--top-k", type=int, default=3, help="results to return for --query")
    parser.add_argument("--mode", choices=("dense", "hybrid"), default="dense",
                        help="retrieval mode for --query (dense baseline vs hybrid)")
    args = parser.parse_args(argv)

    if args.query:
        run_query(args.query, args.top_k, args.mode)
        return 0

    if not Path(args.chunks).is_file():
        parser.error(f"--in not found: {args.chunks}")
    build_index(args.chunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
