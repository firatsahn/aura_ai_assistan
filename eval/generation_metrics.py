"""Generation metrics — LLM-as-judge for faithfulness and answer relevance.

For each answered (non-abstained) question we score the produced answer on two
axes with a separate, stronger judge model (`gpt-4o` by default) than the
generator (`gpt-4o-mini`), to limit self-bias:

- Faithfulness / groundedness: is the answer supported by the retrieved context,
  or does it add unsupported claims? Judge sees answer + context.
- Answer relevance: does the answer actually address the question? Judge sees
  question + answer.

Each judge call returns strict JSON `{"score": 1..5, "reason": "..."}`; scores
are normalised to 0–1 as `(score - 1) / 4` and averaged. Abstained questions
carry no answer to grade, so they are excluded from the means and reported
separately (their correctness is the abstention metric's job).

The judge prompts are Turkish because the answers and corpus are Turkish — this
mirrors the Turkish grounding prompt in `backend/generation/client.py`, while
the surrounding code stays English.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o")

_FAITHFULNESS_PROMPT = (
    "Sen bir değerlendirme hakemisin. Sana bir CEVAP ve bu cevabın dayanması "
    "gereken KAYNAK parçaları verilir. Görevin: cevabın yalnızca kaynaklara "
    "dayanıp dayanmadığını ölçmek.\n"
    "Puanlama (1-5):\n"
    "5 = cevaptaki her bilgi kaynaklarca destekleniyor, uydurma yok.\n"
    "3 = çoğu destekleniyor ama kaynaklarda olmayan en az bir iddia var.\n"
    "1 = cevap büyük ölçüde kaynaklarca desteklenmiyor (halüsinasyon).\n"
    'SADECE şu JSON formatında yanıt ver: {"score": <1-5>, "reason": "<kısa gerekçe>"}'
)

_RELEVANCE_PROMPT = (
    "Sen bir değerlendirme hakemisin. Sana bir SORU ve bir CEVAP verilir. "
    "Görevin: cevabın soruyu gerçekten yanıtlayıp yanıtlamadığını ölçmek "
    "(doğruluğunu değil, soruyla ilgisini ve soruyu karşılamasını).\n"
    "Puanlama (1-5):\n"
    "5 = cevap soruyu tam ve doğrudan karşılıyor.\n"
    "3 = kısmen ilgili veya eksik.\n"
    "1 = soruyla ilgisiz ya da soruyu yanıtlamıyor.\n"
    'SADECE şu JSON formatında yanıt ver: {"score": <1-5>, "reason": "<kısa gerekçe>"}'
)


def _normalise(score: float) -> float:
    """Map a 1–5 judge score onto 0–1, clamped to that range."""
    return (max(1.0, min(5.0, float(score))) - 1.0) / 4.0


def _judge(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """One judge call returning parsed `{"score", "reason"}` (score is 1–5)."""
    client = _client()
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = (response.choices[0].message.content or "{}").strip()
    data = json.loads(raw)
    return {"score": float(data["score"]), "reason": data.get("reason", "")}


def judge_faithfulness(answer: str, context: str) -> dict[str, Any]:
    """Score how well `answer` is grounded in `context` (1–5 + reason)."""
    user = f"KAYNAKLAR:\n{context}\n\nCEVAP:\n{answer}"
    return _judge(_FAITHFULNESS_PROMPT, user)


def judge_relevance(question: str, answer: str) -> dict[str, Any]:
    """Score how well `answer` addresses `question` (1–5 + reason)."""
    user = f"SORU:\n{question}\n\nCEVAP:\n{answer}"
    return _judge(_RELEVANCE_PROMPT, user)


def _context_from_sources(sources: list[dict[str, Any]]) -> str:
    """Rebuild the retrieved context the generator saw, for the faithfulness judge."""
    blocks = []
    for i, s in enumerate(sources, 1):
        blocks.append(f"[Kaynak {i}: {s.get('source_doc')}]\n{s.get('text', '')}")
    return "\n\n".join(blocks)


def evaluate_generation(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Judge faithfulness + relevance over already-produced answer outcomes.

    `outcomes` is the answerable questions run through `answer_question` (one
    pass, shared with abstention scoring): each item has `id`, `question`,
    `answer`, `abstained`, `sources`. Abstained items are skipped in the means.
    """
    per_question: list[dict[str, Any]] = []
    faith_scores: list[float] = []
    rel_scores: list[float] = []
    judged = 0

    for o in outcomes:
        if o["abstained"]:
            per_question.append({"id": o["id"], "skipped": "abstained"})
            continue
        context = _context_from_sources(o["sources"])
        faith = judge_faithfulness(o["answer"], context)
        rel = judge_relevance(o["question"], o["answer"])
        faith_n = _normalise(faith["score"])
        rel_n = _normalise(rel["score"])
        faith_scores.append(faith_n)
        rel_scores.append(rel_n)
        judged += 1
        per_question.append(
            {
                "id": o["id"],
                "faithfulness": faith_n,
                "faithfulness_raw": faith["score"],
                "faithfulness_reason": faith["reason"],
                "relevance": rel_n,
                "relevance_raw": rel["score"],
                "relevance_reason": rel["reason"],
            }
        )

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "judge_model": JUDGE_MODEL,
        "judged": judged,
        "skipped_abstained": len(outcomes) - judged,
        "faithfulness": _mean(faith_scores),
        "answer_relevance": _mean(rel_scores),
        "per_question": per_question,
    }


def _client():
    import openai

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set — needed for the LLM judge "
            f"(model={JUDGE_MODEL}). Add it to .env or run with --retrieval-only."
        )
    return openai.OpenAI()
