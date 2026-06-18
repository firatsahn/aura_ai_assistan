"""Single-command evaluation harness: `python -m eval.run` from the project root.

Runs the four metric families over the golden set and prints a dense-vs-hybrid
table, then writes everything (summary + per-question detail + run config) to
`eval/results.json` for reproducibility. No external eval framework.

Cost control via flags:
- default: retrieval (free) + generation answers + LLM judge + abstention.
- --retrieval-only: only the free, deterministic retrieval metrics (no LLM).
- --no-judge: run answers + abstention but skip the judge calls.

The answerable and abstention questions are run through `answer_question` once
per mode; those outcomes feed both the abstention metric and the generation
judge, so no question is answered twice.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.embedding import EMBEDDING_MODEL
from backend.generation import GENERATION_MODEL
from backend.pipeline import ABSTENTION_MESSAGE, ABSTENTION_THRESHOLD, answer_question

from eval import abstention_metrics, dataset, generation_metrics, retrieval_metrics

MODES = ("dense", "hybrid")
DEFAULT_OUT = Path(__file__).parent / "results.json"


def _run_answers(
    questions: list[dict[str, Any]], mode: str, top_k: int
) -> list[dict[str, Any]]:
    """Answer each question once in `mode`, keeping the fields metrics need.

    The system abstains two ways: the dense score gate (`result["abstained"]`)
    and the grounded prompt emitting the fixed abstention message even when the
    gate let a near-but-irrelevant chunk through. Both are real refusals, so the
    `abstained` we hand to the metrics is the OR of the two; `gate_abstained`
    keeps the raw flag for transparency.
    """
    outcomes = []
    for q in questions:
        result = answer_question(q["question"], top_k=top_k, retrieval_mode=mode)
        gate_abstained = result["abstained"]
        effective = gate_abstained or result["answer"].strip() == ABSTENTION_MESSAGE
        outcomes.append(
            {
                "id": q["id"],
                "question": q["question"],
                "answer": result["answer"],
                "abstained": effective,
                "gate_abstained": gate_abstained,
                "sources": result["sources"],
                "top_score": result["top_score"],
            }
        )
    return outcomes


def evaluate(
    golden: list[dict[str, Any]],
    *,
    retrieval_only: bool,
    run_judge: bool,
    top_k: int,
) -> dict[str, Any]:
    """Compute all requested metrics for both modes and return the result tree."""
    answerable = dataset.answerable(golden)
    abstention_qs = dataset.abstention(golden)

    results: dict[str, Any] = {
        "config": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "embedding_model": EMBEDDING_MODEL,
            "generation_model": GENERATION_MODEL,
            "judge_model": generation_metrics.JUDGE_MODEL,
            "abstention_threshold": ABSTENTION_THRESHOLD,
            "top_k": top_k,
            "n_total": len(golden),
            "n_answerable": len(answerable),
            "n_abstention": len(abstention_qs),
            "retrieval_only": retrieval_only,
            "judge_enabled": run_judge and not retrieval_only,
        },
        "modes": {},
    }

    for mode in MODES:
        print(f"[{mode}] retrieval metrics...", flush=True)
        mode_result: dict[str, Any] = {
            "retrieval": retrieval_metrics.evaluate_retrieval(answerable, mode)
        }

        if not retrieval_only:
            print(f"[{mode}] answering {len(answerable)} + {len(abstention_qs)} questions...", flush=True)
            answerable_outcomes = _run_answers(answerable, mode, top_k)
            abstention_outcomes = _run_answers(abstention_qs, mode, top_k)
            mode_result["abstention"] = abstention_metrics.evaluate_abstention(
                answerable_outcomes, abstention_outcomes
            )
            if run_judge:
                print(f"[{mode}] judging answers ({generation_metrics.JUDGE_MODEL})...", flush=True)
                mode_result["generation"] = generation_metrics.evaluate_generation(
                    answerable_outcomes
                )

        results["modes"][mode] = mode_result

    return results


def _fmt(value: Any) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else str(value)


def print_table(results: dict[str, Any]) -> None:
    """Print the dense-vs-hybrid comparison table to the terminal."""
    dense = results["modes"]["dense"]
    hybrid = results["modes"]["hybrid"]
    rows: list[tuple[str, str, str]] = []

    dr, hr = dense["retrieval"], hybrid["retrieval"]
    for k in sorted(dr["recall_at_k"], key=int):
        rows.append((f"Recall@{k}", _fmt(dr["recall_at_k"][k]), _fmt(hr["recall_at_k"][k])))
    rows.append(("MRR", _fmt(dr["mrr"]), _fmt(hr["mrr"])))

    if "generation" in dense:
        dg, hg = dense["generation"], hybrid["generation"]
        rows.append(("Faithfulness", _fmt(dg["faithfulness"]), _fmt(hg["faithfulness"])))
        rows.append(("Answer relevance", _fmt(dg["answer_relevance"]), _fmt(hg["answer_relevance"])))

    if "abstention" in dense:
        da, ha = dense["abstention"], hybrid["abstention"]
        rows.append((
            "Abstention recall",
            f"{da['correct_abstentions']}/{da['n_abstention']}",
            f"{ha['correct_abstentions']}/{ha['n_abstention']}",
        ))
        rows.append((
            "False abstentions",
            f"{da['false_abstentions']}/{da['n_answerable']}",
            f"{ha['false_abstentions']}/{ha['n_answerable']}",
        ))

    w0 = max(len("Metric"), *(len(r[0]) for r in rows))
    w1 = max(len("Dense (baseline)"), *(len(r[1]) for r in rows))
    w2 = max(len("Hybrid"), *(len(r[2]) for r in rows))
    sep = f"{'-' * w0}-+-{'-' * w1}-+-{'-' * w2}"
    print()
    print(f"{'Metric':<{w0}} | {'Dense (baseline)':<{w1}} | {'Hybrid':<{w2}}")
    print(sep)
    for name, d, h in rows:
        print(f"{name:<{w0}} | {d:<{w1}} | {h:<{w2}}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation harness (golden set).")
    parser.add_argument("--retrieval-only", action="store_true", help="Only retrieval metrics; no LLM calls.")
    parser.add_argument("--no-judge", action="store_true", help="Run answers + abstention but skip the LLM judge.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for retrieval and generation (default 5).")
    parser.add_argument("--golden", type=Path, default=dataset.GOLDEN_SET_PATH, help="Path to golden_set.jsonl.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Where to write results.json.")
    args = parser.parse_args()

    golden = dataset.load_golden_set(args.golden)
    started = time.time()
    results = evaluate(
        golden,
        retrieval_only=args.retrieval_only,
        run_judge=not args.no_judge,
        top_k=args.top_k,
    )
    results["config"]["elapsed_seconds"] = round(time.time() - started, 1)

    print_table(results)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
