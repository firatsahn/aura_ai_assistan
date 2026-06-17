"""Schema: deterministic ids, JSON round-trip, JSONL I/O. Hermetic."""

import json

from backend.ingestion.schema import Chunk, make_chunk_id, read_jsonl, write_jsonl


def test_make_chunk_id_is_deterministic_and_padded():
    assert make_chunk_id("04_hata_kodlari.xlsx", 2) == "04_hata_kodlari::002"
    assert make_chunk_id("a/b/03_led.png", 0) == "03_led::000"
    # same inputs -> same id (idempotent re-ingest)
    assert make_chunk_id("x.pdf", 5) == make_chunk_id("x.pdf", 5)


def test_to_json_preserves_turkish():
    c = Chunk(chunk_id="a::000", text="Yeşil ışık yanıyor", source_doc="a.png",
              doc_type="table", modality="image")
    assert "Yeşil ışık" in c.to_json()  # ensure_ascii=False


def test_json_round_trip_equality():
    c = Chunk(chunk_id="x::000", text="t", source_doc="x.md", doc_type="text",
              modality="markdown", page=1, section="S", metadata={"k": "v"})
    c2 = Chunk.from_dict(json.loads(c.to_json()))
    assert c2 == c


def test_jsonl_round_trip(tmp_path):
    chunks = [
        Chunk("a::000", "t1", "a.md", "text", "markdown"),
        Chunk("a::001", "t2", "a.md", "text", "markdown", page=2, section="S"),
    ]
    out = tmp_path / "nested" / "chunks.jsonl"  # parent dir is created
    assert write_jsonl(chunks, out) == 2
    assert out.exists()
    assert read_jsonl(out) == chunks
