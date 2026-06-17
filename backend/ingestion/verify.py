"""Automated sanity checks over chunks.jsonl (eyeball companion, not a substitute).

    python -m backend.ingestion.verify --in data/chunks.jsonl

HARD checks (deterministic; failing exits non-zero): structural invariants, the
error-code table (doc 04), and the subscription plans (doc 02). SOFT checks
(vision-dependent; reported, never fail the run): LED color rows (03), spec
fields (05), embedded table (08), FAQ sections (07).
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from .schema import Chunk, read_jsonl

EXPECTED_ERROR_CODES = {"E-101", "E-205", "E-212", "E-301", "E-404", "E-500"}
SPEC_KEYWORDS = [
    "işlemci", "RAM", "bellek", "depolama", "boyut", "ağırlık", "mm", "g",
    "GHz", "MHz", "Wi-Fi", "Bluetooth", "Zigbee", "Thread", "güç", "USB",
]


def _by_doc(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    grouped: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        grouped[c.source_doc].append(c)
    return grouped


def _doc(by_doc: dict[str, list[Chunk]], prefix: str) -> list[Chunk]:
    for name, chunks in by_doc.items():
        if name.startswith(prefix):
            return chunks
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", default="data/chunks.jsonl")
    args = parser.parse_args(argv)

    chunks = read_jsonl(args.inp)
    by_doc = _by_doc(chunks)

    print(f"Loaded {len(chunks)} chunks from {len(by_doc)} documents.\n")
    print("Per-document chunk counts:")
    for name in sorted(by_doc):
        modalities = sorted({c.modality for c in by_doc[name]})
        print(f"  {name:48s} {len(by_doc[name]):3d}  [{', '.join(modalities)}]")
    print()

    hard_failures: list[str] = []

    # ---- HARD: structural invariants -------------------------------------- #
    seen_ids: set[str] = set()
    for c in chunks:
        if not c.text.strip():
            hard_failures.append(f"empty text: {c.chunk_id}")
        if not c.source_doc:
            hard_failures.append(f"missing source_doc: {c.chunk_id}")
        if c.chunk_id in seen_ids:
            hard_failures.append(f"duplicate chunk_id: {c.chunk_id}")
        seen_ids.add(c.chunk_id)

    # Every document must produce at least one chunk.
    for name, group in by_doc.items():
        if not group:
            hard_failures.append(f"no chunks for {name}")

    # ---- HARD: doc 04 error codes ----------------------------------------- #
    err = _doc(by_doc, "04")
    err_keys = {c.metadata.get("key", "").strip() for c in err if c.doc_type == "table"}
    missing = EXPECTED_ERROR_CODES - err_keys
    if missing:
        hard_failures.append(f"04: missing error codes {sorted(missing)}")
    for c in err:
        if c.metadata.get("key", "") in EXPECTED_ERROR_CODES and "Önerilen Çözüm" not in c.text:
            hard_failures.append(f"04: chunk {c.chunk_id} ({c.metadata.get('key')}) lacks 'Önerilen Çözüm'")
    print(f"[HARD] 04 error codes: found {sorted(err_keys & EXPECTED_ERROR_CODES)}"
          + (f"  MISSING {sorted(missing)}" if missing else "  (all present, row-per-chunk)"))

    # ---- HARD: doc 02 subscription plans ---------------------------------- #
    sub_text = " ".join(c.text for c in _doc(by_doc, "02"))
    plan_hits = [p for p in ("Plus", "Pro") if p in sub_text]
    has_free = "Free" in sub_text or "Ücretsiz" in sub_text
    if not (has_free and "Plus" in sub_text and "Pro" in sub_text):
        hard_failures.append(f"02: subscription plans incomplete (free={has_free}, hits={plan_hits})")
    print(f"[HARD] 02 plans: free={has_free}, plus={'Plus' in sub_text}, pro={'Pro' in sub_text}")

    # ---- SOFT: doc 03 LED color rows -------------------------------------- #
    led = _doc(by_doc, "03")
    led_rows = [c for c in led if c.doc_type == "table"]
    print(f"\n[SOFT] 03 LED: {len(led)} chunks ({len(led_rows)} table rows)"
          + ("  WARN <5 — re-check vision output" if len(led_rows) < 5 else ""))
    for c in led_rows:
        print(f"        - {c.metadata.get('key', '')[:60]}")

    # ---- SOFT: doc 05 technical specs ------------------------------------- #
    spec_text = " ".join(c.text for c in _doc(by_doc, "05"))
    found_specs = [k for k in SPEC_KEYWORDS if k.lower() in spec_text.lower()]
    print(f"[SOFT] 05 specs: {len(_doc(by_doc, '05'))} chunks; matched fields {found_specs}"
          + ("  WARN none matched" if not found_specs else ""))

    # ---- SOFT: doc 08 embedded table -------------------------------------- #
    table_08 = [c for c in _doc(by_doc, "08") if c.doc_type == "table"]
    print(f"[SOFT] 08 embedded table: {len(table_08)} table-row chunks"
          + ("  (note: table fell back to prose — check find_tables)" if not table_08 else ""))

    # ---- SOFT: doc 07 FAQ sections ---------------------------------------- #
    faq = _doc(by_doc, "07")
    faq_sections = sorted({c.section for c in faq if c.section})
    print(f"[SOFT] 07 FAQ: {len(faq_sections)} distinct sections preserved")

    # ---- summary ---------------------------------------------------------- #
    print("\n" + "=" * 60)
    if hard_failures:
        print(f"FAIL — {len(hard_failures)} hard check(s):")
        for f in hard_failures:
            print(f"  - {f}")
        return 1
    print("PASS — all hard checks passed. Eyeball the SOFT lines + chunks.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
