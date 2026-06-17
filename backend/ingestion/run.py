"""CLI: ingest every supported file in doc/ into a single chunks.jsonl.

    python -m backend.ingestion.run --doc-dir doc --out data/chunks.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .router import load_file, supported
from .schema import write_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-dir", default="doc", help="directory of source documents")
    parser.add_argument("--out", default="data/chunks.jsonl", help="output JSONL path")
    args = parser.parse_args(argv)

    doc_dir = Path(args.doc_dir)
    if not doc_dir.is_dir():
        parser.error(f"--doc-dir not found: {doc_dir}")

    files = sorted(p for p in doc_dir.iterdir() if p.is_file() and supported(p))
    if not files:
        parser.error(f"no supported files in {doc_dir}")

    all_chunks = []
    print(f"Ingesting {len(files)} files from {doc_dir}/")
    for path in files:
        chunks = load_file(path)
        all_chunks.extend(chunks)
        print(f"  {path.name:48s} -> {len(chunks):3d} chunks")

    count = write_jsonl(all_chunks, args.out)
    print(f"\nWrote {count} chunks to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
