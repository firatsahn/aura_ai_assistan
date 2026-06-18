"""Golden-set loader and the answerable / abstention split.

One source of truth for reading `golden_set.jsonl` so every metric module agrees
on what counts as an answerable question (has `expected_sources`) versus an
abstention probe (`expected_answer` is null, no sources). The split is by data,
not by the `category` field, so the metrics stay correct even if categories are
relabelled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"


def load_golden_set(path: Path | str = GOLDEN_SET_PATH) -> list[dict[str, Any]]:
    """Read the JSONL golden set into a list of question records, in order."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def answerable(golden: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Questions the corpus should answer: those with at least one expected source."""
    return [q for q in golden if q.get("expected_sources")]


def abstention(golden: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Out-of-corpus probes: no expected answer and no expected sources."""
    return [q for q in golden if not q.get("expected_sources") and q.get("expected_answer") is None]
