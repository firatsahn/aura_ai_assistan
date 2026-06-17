"""Chunker: Markdown parsing, table->rows, prose splitting, ids. Hermetic."""

from backend.ingestion import chunker
from backend.ingestion.chunker import (
    ProseBlock,
    TableBlock,
    blocks_to_chunks,
    parse_markdown_blocks,
)


# --------------------------------------------------------------------------- #
# Markdown parsing
# --------------------------------------------------------------------------- #
def test_parse_markdown_sections():
    md = "# Title\n\nintro para.\n\n## Section A\n\nbody a.\n"
    proses = [b for b in parse_markdown_blocks(md) if isinstance(b, ProseBlock)]
    assert proses[0].section == "Title" and "intro para." in proses[0].text
    assert proses[1].section == "Section A" and "body a." in proses[1].text


def test_parse_markdown_table():
    md = "## T\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n"
    tables = [b for b in parse_markdown_blocks(md) if isinstance(b, TableBlock)]
    assert len(tables) == 1
    assert tables[0].header == ["A", "B"]
    assert tables[0].rows == [["1", "2"], ["3", "4"]]
    assert tables[0].section == "T"


# --------------------------------------------------------------------------- #
# Table -> one row per chunk (header-prefixed, never split)
# --------------------------------------------------------------------------- #
def test_table_block_yields_one_chunk_per_row():
    tb = TableBlock(header=["Kod", "Çözüm"], rows=[["E1", "do x"], ["E2", "do y"]],
                    section="S", title="S")
    chunks = blocks_to_chunks([tb], source_doc="04_x.xlsx", modality="spreadsheet")
    assert len(chunks) == 2
    assert [c.chunk_id for c in chunks] == ["04_x::000", "04_x::001"]
    assert all(c.doc_type == "table" for c in chunks)
    assert chunks[0].text.startswith("S")            # title prefix
    assert "Kod: E1" in chunks[0].text and "Çözüm: do x" in chunks[0].text
    assert chunks[0].metadata["key"] == "E1"


def test_block_modality_override_wins():
    tb = TableBlock(header=["A"], rows=[["1"]], modality="pdf_text")
    chunks = blocks_to_chunks([tb], source_doc="x.pdf", modality="pdf_scanned")
    assert chunks[0].modality == "pdf_text"  # per-block override beats default


# --------------------------------------------------------------------------- #
# Prose splitting
# --------------------------------------------------------------------------- #
def test_short_prose_is_single_chunk():
    chunks = blocks_to_chunks([ProseBlock(text="short text", section="Sec")],
                              source_doc="01_a.md", modality="markdown")
    assert len(chunks) == 1
    assert chunks[0].doc_type == "text" and chunks[0].section == "Sec"
    assert chunks[0].chunk_id == "01_a::000"


def test_long_prose_splits_and_covers_all_content():
    text = "\n\n".join(f"para{i} " + "x" * 400 for i in range(12))
    pieces = chunker._split_prose(text, budget=1000, overlap=100)
    assert len(pieces) > 1
    assert all(p.strip() for p in pieces)
    joined = " ".join(pieces)
    for i in range(12):
        assert f"para{i}" in joined  # no content dropped


def test_ids_are_sequential_across_mixed_blocks():
    blocks = [
        ProseBlock(text="intro", section="A"),
        TableBlock(header=["H"], rows=[["1"], ["2"]], section="B"),
        ProseBlock(text="outro", section="C"),
    ]
    chunks = blocks_to_chunks(blocks, source_doc="06_x.pdf", modality="pdf_text")
    assert [c.chunk_id for c in chunks] == [
        "06_x::000", "06_x::001", "06_x::002", "06_x::003",
    ]
