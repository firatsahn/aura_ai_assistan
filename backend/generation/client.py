"""OpenAI generation client: retrieved chunks + question -> grounded answer.

One public function, `generate(question, hits)`, mirroring `embed()` in
`backend/embedding/client.py`: a single module so the provider can change in one
place. It takes the top-k chunks from retrieval, lays them out as a numbered,
source-tagged context, and asks the model to answer *only* from that context —
citing its source and saying it does not know when the context does not cover
the question. This grounding-or-abstain instruction is what keeps the system
from hallucinating over a small, closed corpus.

The default model is OpenAI `gpt-4o-mini`. The embedding side already runs on
OpenAI, so generation here reuses the same provider and key (one surface). The
locked target was Claude `claude-opus-4-8` (see DECISIONS.tr.md §1.4); swapping
back is a one-file change behind this `generate()` boundary. Provider config is
read from env (`GENERATION_MODEL`, `OPENAI_API_KEY`).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from backend.vectorstore import SearchHit

load_dotenv()  # load credentials / config from project .env if present

GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gpt-4o-mini")

# Corpus is Turkish; answer in its language and cite provenance. The hard rule
# is grounding: anything not in the supplied context must trigger abstention,
# never a guess.
_SYSTEM_PROMPT = (
    "Sen Aura akıllı ev sisteminin destek asistanısın. Sana yalnızca bir soru ve "
    "o soruyla ilgili bulunmuş doküman parçaları (bağlam) verilir.\n"
    "Kurallar:\n"
    "1. Cevabını SADECE verilen bağlama dayandır. Bağlamda olmayan hiçbir bilgiyi "
    "ekleme, tahmin etme veya uydurma.\n"
    "2. Bağlam soruyu yanıtlamaya yetmiyorsa, tam olarak şunu yaz: "
    "'Bu bilgi tabanında bunu bulamadım.'\n"
    "3. Cevabın sonunda kullandığın kaynağı belirt (ör. 'Kaynak: 03_led_durum_gostergeleri').\n"
    "4. Korpusun dilinde, yani Türkçe yanıtla. Kısa ve net ol."
)


def generate(question: str, hits: list[SearchHit]) -> str:
    """Answer `question` grounded in `hits`, citing sources or abstaining."""
    context = _format_context(hits)
    user_prompt = (
        f"Bağlam:\n{context}\n\n"
        f"Soru: {question}\n\n"
        "Yukarıdaki kurallara uyarak cevapla."
    )

    client = _client()
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,  # deterministic, grounded — no creative drift
    )
    return (response.choices[0].message.content or "").strip()


def _format_context(hits: list[SearchHit]) -> str:
    """Lay out hits as numbered, source-tagged blocks for the prompt."""
    blocks: list[str] = []
    for i, hit in enumerate(hits, 1):
        section = hit.payload.get("section")
        label = f"Kaynak {i}: {hit.source_doc}"
        if section:
            label += f" — {section}"
        blocks.append(f"[{label}]\n{hit.text}")
    return "\n\n".join(blocks)


def _client():
    import openai

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set — needed for generation "
            f"(model={GENERATION_MODEL}). Add it to .env."
        )
    return openai.OpenAI()
