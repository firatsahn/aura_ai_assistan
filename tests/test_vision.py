"""Vision: provider resolution + cache behavior. No live API calls (monkeypatched)."""

from backend.ingestion import vision


def test_resolve_provider_precedence(monkeypatch):
    for var in ("VISION_PROVIDER", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    assert vision._resolve_provider() == "anthropic"          # default when nothing set
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert vision._resolve_provider() == "gemini"             # only google key -> gemini
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert vision._resolve_provider() == "anthropic"          # anthropic wins when both set
    monkeypatch.setenv("VISION_PROVIDER", "gemini")
    assert vision._resolve_provider() == "gemini"             # explicit override wins


def test_cache_hit_does_not_call_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "CACHE_DIR", tmp_path)
    (tmp_path / "doc1.md").write_text("# cached markdown", encoding="utf-8")

    calls = []
    monkeypatch.setattr(vision, "_extract_gemini", lambda *a, **k: calls.append("g"))
    monkeypatch.setattr(vision, "_extract_anthropic", lambda *a, **k: calls.append("a"))

    out = vision.extract(b"ignored", media_type="image/png", doc_name="doc1.png")
    assert out == "# cached markdown"
    assert calls == []  # provider never invoked on a cache hit


def test_cache_miss_calls_provider_and_writes_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("VISION_PROVIDER", "gemini")
    monkeypatch.setattr(vision, "_extract_gemini", lambda b, m, d: "# fresh transcription")

    out = vision.extract(b"x", media_type="image/png", doc_name="doc2.png")
    assert out == "# fresh transcription"
    assert (tmp_path / "doc2.md").read_text(encoding="utf-8") == "# fresh transcription"
