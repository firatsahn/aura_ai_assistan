"""Loader tests with self-contained fixtures (no corpus, no API)."""

import openpyxl

from backend.ingestion.loaders import markdown, pdf, spreadsheet


def test_markdown_loader(tmp_path):
    p = tmp_path / "01_x.md"
    p.write_text("# T\n\nhello\n\n## A\n\nworld\n", encoding="utf-8")
    chunks = markdown.load(p)
    assert chunks
    assert all(c.modality == "markdown" for c in chunks)
    assert "A" in {c.section for c in chunks}


def test_spreadsheet_loader_row_per_chunk(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Hata Kodları"
    ws.append(["Hata Kodu", "Çözüm"])
    ws.append(["E-101", "fix x"])
    ws.append(["E-205", "fix y"])
    ws.append([None, None])  # blank row should be dropped
    p = tmp_path / "04_x.xlsx"
    wb.save(p)

    chunks = spreadsheet.load(p)
    assert len(chunks) == 2
    assert {c.metadata["key"] for c in chunks} == {"E-101", "E-205"}
    assert all(c.doc_type == "table" and c.modality == "spreadsheet" for c in chunks)
    assert "Hata Kodu: E-101" in chunks[0].text and "Çözüm: fix x" in chunks[0].text


def test_pdf_native_detects_heading(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    # ASCII only — base-14 fonts don't carry Turkish glyphs; we test detection logic.
    page.insert_text((72, 72), "1. Genel Bakis", fontsize=14, fontname="hebo")  # heading
    for i, y in enumerate((110, 130, 150)):  # 3 body lines -> body size = 9 (the mode)
        page.insert_text((72, y), f"Govde paragrafi satir {i} aciklama metni.",
                         fontsize=9, fontname="helv")
    p = tmp_path / "06_x.pdf"
    doc.save(p)
    doc.close()

    chunks = pdf.load(p)
    assert chunks
    assert all(c.modality == "pdf_text" for c in chunks)
    assert any(c.section == "1. Genel Bakis" for c in chunks)
