"""Abstention metrics — does the system refuse exactly when it should?

A corpus-specific, high-value check that standard RAG evals skip: the brief
requires the system to abstain rather than guess when it can't answer reliably.
Two directions, both read off `answer_question`'s `abstained` flag:

- Abstention recall: of the out-of-corpus probes (q41–q45), how many correctly
  abstained — reported as "X/5".
- False abstentions: of the answerable questions, how many wrongly abstained.
  Each false abstention is a question the corpus could answer but the system
  refused — a usability cost, the opposite failure of hallucination.

This module does no model calls; it consumes the same `answer_question` outcomes
the run collects once per mode.
"""

from __future__ import annotations

from typing import Any


def evaluate_abstention(
    answerable_outcomes: list[dict[str, Any]],
    abstention_outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Correct abstentions on probes + false abstentions on answerable questions."""
    correct_abstentions = sum(1 for o in abstention_outcomes if o["abstained"])
    false_abstentions = sum(1 for o in answerable_outcomes if o["abstained"])

    n_abstention = len(abstention_outcomes)
    n_answerable = len(answerable_outcomes)

    return {
        "correct_abstentions": correct_abstentions,
        "n_abstention": n_abstention,
        "abstention_recall": correct_abstentions / n_abstention if n_abstention else 0.0,
        "false_abstentions": false_abstentions,
        "n_answerable": n_answerable,
        # Of everything the system chose to answer, what fraction was warranted.
        "answer_precision": (
            (n_answerable - false_abstentions)
            / (n_answerable - false_abstentions + (n_abstention - correct_abstentions))
            if (n_answerable - false_abstentions + (n_abstention - correct_abstentions))
            else 0.0
        ),
        "false_abstained_ids": [o["id"] for o in answerable_outcomes if o["abstained"]],
        "wrongly_answered_ids": [o["id"] for o in abstention_outcomes if not o["abstained"]],
    }
