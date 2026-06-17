"""Common chunk data model and JSONL (de)serialization.

Every loader returns a list of `Chunk`. This is the single contract the rest of
the pipeline depends on, fixed up front so all loaders conform to it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Chunk:
    """One retrievable unit of text with its provenance metadata."""

    chunk_id: str            # deterministic, e.g. "04_hata_kodlari::002"
    text: str                # content to embed (Turkish, kept verbatim)
    source_doc: str          # e.g. "03_led_durum_gostergeleri.png"
    doc_type: str            # "text" | "table"
    modality: str            # markdown | pdf_text | pdf_scanned | image | spreadsheet
    page: int | None = None  # 1-based PDF page, else None
    section: str | None = None  # nearest heading, e.g. "Abonelik Planları"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        # ensure_ascii=False keeps Turkish readable when eyeballing chunks.jsonl
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(**d)


def make_chunk_id(source_doc: str, index: int) -> str:
    """Deterministic id: same input -> same id (idempotent re-ingest / diffable)."""
    return f"{Path(source_doc).stem}::{index:03d}"


def write_jsonl(chunks: Iterable[Chunk], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.to_json() + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks
