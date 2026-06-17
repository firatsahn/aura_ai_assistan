"""Integration tests against the real corpus.

The corpus (doc/) and the vision cache (.cache/) are gitignored, so these tests
skip cleanly when those files are absent (e.g. in CI / a fresh clone) and run
locally where they exist. They assert the brief's key facts end to end.
"""

from pathlib import Path

import pytest

from backend.ingestion.router import load_file

DOC = Path("doc")
CACHE = Path("backend/ingestion/.cache")


def _need(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            pytest.skip(f"required file absent: {p}")


def test_error_codes_each_a_row_chunk():
    f = DOC / "04_hata_kodlari.xlsx"
    _need(f)
    keys = {c.metadata.get("key") for c in load_file(f)}
    assert {"E-101", "E-205", "E-212", "E-301", "E-404", "E-500"} <= keys


def test_subscription_plans_present():
    f = DOC / "02_abonelik_ve_iade_politikasi.pdf"
    _need(f)
    text = " ".join(c.text for c in load_file(f))
    assert "Plus" in text and "Pro" in text and ("Free" in text or "Ücretsiz" in text)


def test_privacy_doc_embedded_table_extracted():
    f = DOC / "08_gizlilik_ve_veri_politikasi.pdf"
    _need(f)
    chunks = load_file(f)
    assert any(c.doc_type == "table" for c in chunks)  # find_tables caught the table


def test_faq_question_sections_preserved():
    f = DOC / "07_sorun_giderme_sss.pdf"
    _need(f)
    sections = {c.section for c in load_file(f) if c.section}
    assert len(sections) >= 10


def test_led_image_all_color_rows():
    png = DOC / "03_led_durum_gostergeleri.png"
    _need(png, CACHE / "03_led_durum_gostergeleri.md")  # uses cached transcript, no API
    rows = [c for c in load_file(png) if c.doc_type == "table"]
    assert len(rows) >= 5  # expected ~7 LED colours


def test_scanned_specs_extracted():
    pdf_path = DOC / "05_teknik_ozellikler_scan.pdf"
    _need(pdf_path, CACHE / "05_teknik_ozellikler_scan_p1.md")  # cached transcript
    chunks = load_file(pdf_path)
    assert chunks and all(c.modality == "pdf_scanned" for c in chunks)
    text = " ".join(c.text for c in chunks).lower()
    assert any(k in text for k in ("ram", "mm", "ghz", "depolama"))
