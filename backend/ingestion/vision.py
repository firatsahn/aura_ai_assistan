"""Shared vision: image bytes -> faithful Markdown, with disk cache.

Provider-agnostic. Defaults to Claude (claude-opus-4-8); uses Gemini when that is
selected or when only a Google key is present. The provider is chosen by the
VISION_PROVIDER env var, otherwise auto-detected from available credentials.

Used by the image loader (doc 03) and the scanned-page path of the PDF loader
(doc 05). Output is cached to .cache/<name>.md so re-running ingestion is free
and deterministic — only the first run hits the API.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # load credentials / provider config from project .env if present

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
ANTHROPIC_MODEL = os.environ.get("VISION_MODEL", "claude-opus-4-8")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
MAX_TOKENS = 8000

_PROMPT = """You are a precise document transcription tool. Transcribe the attached image to faithful Markdown.

Rules:
- Output ONLY the transcription. No preamble, no commentary, no surrounding code fences.
- Preserve the original language exactly (the document is in Turkish). Do NOT translate.
- Render every table as a GitHub-flavored Markdown table. Include EVERY row and column — never omit, merge, or summarize rows.
- For colors or status indicators shown visually, write a textual description of the color/state in the relevant cell (e.g. "Yeşil (sabit)", "Kırmızı (yanıp sönüyor)").
- Preserve headings using # and ## according to their visual hierarchy.
- If the page is a specifications sheet with no other structure, render it as a two-column table of field and value.
"""


def _resolve_provider() -> str:
    explicit = os.environ.get("VISION_PROVIDER", "").strip().lower()
    if explicit in ("anthropic", "gemini"):
        return explicit
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return "anthropic"  # default; raises a helpful error below if no creds


def extract(image_bytes: bytes, *, media_type: str, doc_name: str, use_cache: bool = True) -> str:
    """Transcribe an image to Markdown. `doc_name` keys the on-disk cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{Path(doc_name).stem}.md"
    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    provider = _resolve_provider()
    if provider == "gemini":
        markdown = _extract_gemini(image_bytes, media_type, doc_name)
    else:
        markdown = _extract_anthropic(image_bytes, media_type, doc_name)

    cache_path.write_text(markdown, encoding="utf-8")
    return markdown


def _extract_anthropic(image_bytes: bytes, media_type: str, doc_name: str) -> str:
    import anthropic

    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": b64},
                        },
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ],
        )
    except (anthropic.AuthenticationError, TypeError) as exc:
        # AuthenticationError: bad key. TypeError: SDK could not resolve any credential.
        raise RuntimeError(
            f"Anthropic auth is not configured for the vision step ({doc_name}): {exc}. "
            "Set ANTHROPIC_API_KEY in .env, or set VISION_PROVIDER=gemini with a GEMINI_API_KEY."
        ) from exc

    return "".join(b.text for b in response.content if b.type == "text").strip()


def _extract_gemini(image_bytes: bytes, media_type: str, doc_name: str) -> str:
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            f"GEMINI_API_KEY is not set — needed for the vision step ({doc_name}). Add it to .env."
        )

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=media_type), _PROMPT],
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError(f"Gemini returned an empty transcription for {doc_name} (model={GEMINI_MODEL}).")
    return text
