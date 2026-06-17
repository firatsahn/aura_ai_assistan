"""Type-aware chunking.

Loaders extract content into lightweight `Block`s (prose or table). This module
turns blocks into `Chunk`s with deterministic, sequential ids:

  - TableBlock -> one chunk per row (header-prefixed, self-contained). Never split
    a row; splitting would break its integrity.
  - ProseBlock -> structure-aware split that respects section boundaries, targets
    a character budget (~500-800 tokens), and leaves a small overlap.

`parse_markdown_blocks` is the shared Markdown -> Block parser, reused by the
markdown loader (doc 01) and by the vision output of the image / scanned-PDF
loaders (docs 03, 05).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Union

from .schema import Chunk, make_chunk_id

# ~500-800 tokens of Turkish prose. We size by characters (not tiktoken: it's the
# wrong tokenizer for Claude, and Step 1 stays API-free for the text path).
DEFAULT_CHAR_BUDGET = 2800
DEFAULT_OVERLAP = 300


@dataclass
class ProseBlock:
    text: str
    section: str | None = None
    page: int | None = None
    modality: str | None = None  # overrides the loader default when set


@dataclass
class TableBlock:
    header: list[str]
    rows: list[list[str]]
    section: str | None = None
    page: int | None = None
    title: str | None = None  # label prefixed to each row chunk, e.g. "LED Durum Göstergeleri"
    modality: str | None = None


Block = Union[ProseBlock, TableBlock]


# --------------------------------------------------------------------------- #
# Blocks -> Chunks
# --------------------------------------------------------------------------- #
def blocks_to_chunks(
    blocks: list[Block],
    *,
    source_doc: str,
    modality: str,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for block in blocks:
        mod = block.modality or modality
        if isinstance(block, TableBlock):
            for row_index, row in enumerate(block.rows):
                text = _render_row(block.header, row, block.title)
                if not text.strip():
                    continue
                key = row[0].strip() if row else ""
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(source_doc, index),
                        text=text,
                        source_doc=source_doc,
                        doc_type="table",
                        modality=mod,
                        page=block.page,
                        section=block.section,
                        metadata={"row_index": row_index, "key": key, "columns": block.header},
                    )
                )
                index += 1
        else:
            for piece in _split_prose(block.text, char_budget, overlap):
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(source_doc, index),
                        text=piece,
                        source_doc=source_doc,
                        doc_type="text",
                        modality=mod,
                        page=block.page,
                        section=block.section,
                        metadata={},
                    )
                )
                index += 1
    return chunks


def _render_row(header: list[str], row: list[str], title: str | None) -> str:
    """Render a table row as self-contained `Column: value` lines, header-prefixed."""
    lines: list[str] = []
    if title:
        lines.append(title)
    for i, cell in enumerate(row):
        col = header[i] if i < len(header) else f"col{i + 1}"
        cell = (cell or "").strip()
        if not col and not cell:
            continue
        lines.append(f"{col}: {cell}".strip())
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Prose splitting (paragraph-greedy, char budget, tail overlap)
# --------------------------------------------------------------------------- #
def _split_prose(text: str, budget: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= budget:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > budget:
            if current.strip():
                pieces.append(current.strip())
                current = ""
            pieces.extend(_hard_split(para, budget))
            continue
        candidate = para if not current else current + "\n\n" + para
        if len(candidate) > budget and current:
            pieces.append(current.strip())
            tail = _tail(current, overlap)
            current = (tail + "\n\n" + para) if tail else para
        else:
            current = candidate
    if current.strip():
        pieces.append(current.strip())
    return pieces


def _hard_split(paragraph: str, budget: int) -> list[str]:
    """Split an over-budget paragraph at sentence boundaries (char-slice last resort)."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    out: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > budget:
            if current.strip():
                out.append(current.strip())
                current = ""
            for i in range(0, len(sentence), budget):
                out.append(sentence[i : i + budget].strip())
            continue
        candidate = sentence if not current else current + " " + sentence
        if len(candidate) > budget and current:
            out.append(current.strip())
            current = sentence
        else:
            current = candidate
    if current.strip():
        out.append(current.strip())
    return out


def _tail(text: str, n: int) -> str:
    """Last ~n chars, cut at a word boundary, for inter-chunk overlap."""
    if n <= 0 or len(text) <= n:
        return text.strip()
    fragment = text[-n:]
    space = fragment.find(" ")
    return fragment[space + 1 :].strip() if space != -1 else fragment.strip()


# --------------------------------------------------------------------------- #
# Shared Markdown -> Block parser (doc 01, and vision output for 03 / 05)
# --------------------------------------------------------------------------- #
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def parse_markdown_blocks(md: str, *, default_page: int | None = None) -> list[Block]:
    lines = md.splitlines()
    blocks: list[Block] = []
    section: str | None = None
    prose: list[str] = []

    def flush_prose() -> None:
        nonlocal prose
        text = "\n".join(prose).strip()
        if text:
            blocks.append(ProseBlock(text=text, section=section, page=default_page))
        prose = []

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        heading = _HEADING_RE.match(stripped)
        if heading:
            flush_prose()
            section = heading.group(2).strip()
            i += 1
            continue

        if _is_table_row(stripped) and i + 1 < len(lines) and _is_separator(lines[i + 1].strip()):
            flush_prose()
            table_lines = []
            while i < len(lines) and _is_table_row(lines[i].strip()):
                table_lines.append(lines[i].strip())
                i += 1
            header, rows = _parse_md_table(table_lines)
            if rows:
                blocks.append(
                    TableBlock(
                        header=header,
                        rows=rows,
                        section=section,
                        page=default_page,
                        title=section,
                    )
                )
            continue

        prose.append(raw)
        i += 1

    flush_prose()
    return blocks


def _is_table_row(s: str) -> bool:
    return s.startswith("|") and s.count("|") >= 2


def _is_separator(s: str) -> bool:
    return "-" in s and re.fullmatch(r"\|?[\s:\-|]+\|?", s) is not None


def _split_md_row(s: str) -> list[str]:
    s = s.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _parse_md_table(table_lines: list[str]) -> tuple[list[str], list[list[str]]]:
    header = _split_md_row(table_lines[0])
    rows = [_split_md_row(r) for r in table_lines[2:]]  # skip header + separator
    return header, rows
