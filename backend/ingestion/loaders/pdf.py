"""PDF loader (docs 02, 05, 06, 07, 08).

Routes by content, not by filename: both native and scanned docs are `.pdf`, so
the loader probes each page's text layer. A page with negligible extractable
text is treated as scanned -> rasterized -> sent to Claude vision. A native page
is parsed with font metadata to recover heading boundaries, and tables are
extracted structurally (kept separate from prose).
"""

from __future__ import annotations

import statistics
from pathlib import Path

import fitz  # PyMuPDF

from .. import vision
from ..chunker import Block, ProseBlock, TableBlock, blocks_to_chunks, parse_markdown_blocks
from ..schema import Chunk

# A native page yields hundreds of chars; a scanned page yields ~0. 20 is a safe gate.
SCANNED_TEXT_THRESHOLD = 20
RASTER_DPI = 200
MAX_HEADING_LEN = 90  # a heading is short; longer bold lines are emphasized prose


def load(path: str | Path) -> list[Chunk]:
    path = Path(path)
    doc = fitz.open(path)
    blocks: list[Block] = []
    saw_scanned = saw_native = False
    try:
        for pno in range(doc.page_count):
            page = doc[pno]
            if len(page.get_text("text").strip()) < SCANNED_TEXT_THRESHOLD:
                saw_scanned = True
                blocks.extend(_scanned_page_blocks(page, pno + 1, path.stem))
            else:
                saw_native = True
                blocks.extend(_native_page_blocks(page, pno + 1))
    finally:
        doc.close()

    # Per-block modality (set below) keeps mixed PDFs correct; this is just the default.
    default_modality = "pdf_scanned" if (saw_scanned and not saw_native) else "pdf_text"
    return blocks_to_chunks(blocks, source_doc=path.name, modality=default_modality)


def _scanned_page_blocks(page: "fitz.Page", page_no: int, stem: str) -> list[Block]:
    pix = page.get_pixmap(dpi=RASTER_DPI)
    png = pix.tobytes("png")
    markdown = vision.extract(png, media_type="image/png", doc_name=f"{stem}_p{page_no}.png")
    blocks = parse_markdown_blocks(markdown, default_page=page_no)
    for b in blocks:
        b.modality = "pdf_scanned"
    return blocks


def _native_page_blocks(page: "fitz.Page", page_no: int) -> list[Block]:
    tables = _find_tables(page)  # [(rows, bbox), ...]
    table_rects = [rect for _, rect in tables]
    body_size = _body_size(page)

    blocks: list[Block] = []
    headings: list[tuple[float, str]] = []  # (y0, heading) to map tables to sections
    section: str | None = None
    paragraphs: list[str] = []

    def flush() -> None:
        nonlocal paragraphs
        text = "\n\n".join(paragraphs).strip()
        if text:
            blocks.append(ProseBlock(text=text, section=section, page=page_no, modality="pdf_text"))
        paragraphs = []

    for tblock in page.get_text("dict")["blocks"]:
        if tblock.get("type") != 0:  # skip image blocks
            continue
        rect = fitz.Rect(tblock["bbox"])
        if _center_in_any(rect, table_rects):
            continue  # table interior handled as a TableBlock

        parts: list[tuple[str, float, bool]] = []
        for line in tblock.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s["text"] for s in spans).strip()
            if not text or _center_in_any(fitz.Rect(line["bbox"]), table_rects):
                continue
            size = round(max(s["size"] for s in spans), 1)
            bold = any(s["flags"] & 16 for s in spans)
            parts.append((text, size, bold))
        if not parts:
            continue

        joined = " ".join(p[0] for p in parts)
        max_size = max(p[1] for p in parts)
        any_bold = any(p[2] for p in parts)
        if _is_heading(joined, max_size, any_bold, body_size):
            flush()
            section = joined
            headings.append((rect.y0, joined))
        else:
            paragraphs.append(joined)
    flush()

    for rows, rect in tables:
        sec = _section_for_y(headings, rect.y0)
        blocks.append(
            TableBlock(
                header=rows[0],
                rows=rows[1:],
                section=sec,
                page=page_no,
                title=sec,
                modality="pdf_text",
            )
        )
    return blocks


def _find_tables(page: "fitz.Page") -> list[tuple[list[list[str]], "fitz.Rect"]]:
    """Return (cleaned_rows, bbox) for each detected table with a header + >=1 data row."""
    try:
        found = list(page.find_tables().tables)
    except Exception:
        return []
    out: list[tuple[list[list[str]], "fitz.Rect"]] = []
    for t in found:
        rows = [[(c or "").strip() for c in row] for row in t.extract()]
        rows = [r for r in rows if any(r)]
        if len(rows) >= 2:
            out.append((rows, fitz.Rect(t.bbox)))
    return out


def _body_size(page: "fitz.Page") -> float:
    sizes: list[float] = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                if span["text"].strip():
                    sizes.append(round(span["size"], 1))
    if not sizes:
        return 10.0
    try:
        return statistics.mode(sizes)
    except statistics.StatisticsError:
        return min(sizes)


def _is_heading(text: str, size: float, bold: bool, body: float) -> bool:
    if len(text) > MAX_HEADING_LEN:
        return False
    return size >= body + 2.0 or (bold and size >= body + 1.0)


def _center_in_any(rect: "fitz.Rect", table_rects: list["fitz.Rect"]) -> bool:
    cx = (rect.x0 + rect.x1) / 2
    cy = (rect.y0 + rect.y1) / 2
    point = fitz.Point(cx, cy)
    return any(tr.contains(point) for tr in table_rects)


def _section_for_y(headings: list[tuple[float, str]], y: float) -> str | None:
    section = None
    for hy, htext in headings:
        if hy <= y:
            section = htext
        else:
            break
    return section
