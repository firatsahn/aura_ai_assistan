"""Image loader (doc 03) — read via Claude vision, then chunk the Markdown."""

from __future__ import annotations

from pathlib import Path

from .. import vision
from ..chunker import blocks_to_chunks, parse_markdown_blocks
from ..schema import Chunk

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def load(path: str | Path) -> list[Chunk]:
    path = Path(path)
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/png")
    markdown = vision.extract(path.read_bytes(), media_type=media_type, doc_name=path.name)
    blocks = parse_markdown_blocks(markdown)
    return blocks_to_chunks(blocks, source_doc=path.name, modality="image")
