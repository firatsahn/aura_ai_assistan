"""Markdown loader (doc 01) — read directly, split on heading structure."""

from __future__ import annotations

from pathlib import Path

from ..chunker import blocks_to_chunks, parse_markdown_blocks
from ..schema import Chunk


def load(path: str | Path) -> list[Chunk]:
    path = Path(path)
    md = path.read_text(encoding="utf-8")
    blocks = parse_markdown_blocks(md)
    return blocks_to_chunks(blocks, source_doc=path.name, modality="markdown")
