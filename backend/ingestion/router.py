"""Route a file to the right loader by extension.

PDFs are deliberately a single loader: native vs scanned cannot be told from the
extension, so `loaders.pdf` decides per page (text-layer probe).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .loaders import image, markdown, pdf, spreadsheet
from .schema import Chunk

_DISPATCH: dict[str, Callable[[Path], list[Chunk]]] = {
    ".md": markdown.load,
    ".markdown": markdown.load,
    ".pdf": pdf.load,
    ".png": image.load,
    ".jpg": image.load,
    ".jpeg": image.load,
    ".gif": image.load,
    ".webp": image.load,
    ".xlsx": spreadsheet.load,
    ".xlsm": spreadsheet.load,
}


def supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _DISPATCH


def load_file(path: str | Path) -> list[Chunk]:
    path = Path(path)
    loader = _DISPATCH.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"No loader for extension {path.suffix!r}: {path.name}")
    return loader(path)
