"""Spreadsheet loader (doc 04) — row-wise, header preserved (one row = one chunk)."""

from __future__ import annotations

from pathlib import Path

import openpyxl

from ..chunker import TableBlock, blocks_to_chunks
from ..schema import Chunk


def load(path: str | Path) -> list[Chunk]:
    path = Path(path)
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    blocks: list[TableBlock] = []
    try:
        for sheet in workbook.worksheets:
            rows = [
                [("" if cell is None else str(cell)).strip() for cell in row]
                for row in sheet.iter_rows(values_only=True)
            ]
            rows = [r for r in rows if any(cell for cell in r)]  # drop blank rows
            if len(rows) < 2:
                continue
            header, data = rows[0], rows[1:]
            # No title prefix: the column names already make each row self-contained,
            # and sheet names ("Sheet1") would be noise. The sheet name is kept as `section`.
            blocks.append(TableBlock(header=header, rows=data, section=sheet.title, title=None))
    finally:
        workbook.close()
    return blocks_to_chunks(blocks, source_doc=path.name, modality="spreadsheet")
