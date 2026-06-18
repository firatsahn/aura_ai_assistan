"""Pure-math eval metrics: Recall@k, MRR, abstention. No Qdrant / network / LLM.

Exercises the scoring functions on synthetic ranked source lists so the metric
arithmetic (the part the README table rests on) is verified deterministically,
independent of retrieval quality.
"""

from eval.abstention_metrics import evaluate_abstention
from eval.retrieval_metrics import recall_at_k, reciprocal_rank


def test_recall_hits_within_k():
    ranked = ["a.md", "b.md", "c.md"]
    assert recall_at_k(ranked, ["c.md"], 3) == 1.0
    assert recall_at_k(ranked, ["c.md"], 2) == 0.0  # c.md is at rank 3, outside k=2


def test_recall_any_expected_counts_as_hit():
    # Multi-source question: any one expected source in top-k is a hit.
    ranked = ["x.md", "b.md"]
    assert recall_at_k(ranked, ["a.md", "b.md"], 2) == 1.0
    assert recall_at_k(ranked, ["a.md", "z.md"], 2) == 0.0


def test_reciprocal_rank_uses_first_expected_position():
    ranked = ["a.md", "b.md", "c.md"]
    assert reciprocal_rank(ranked, ["b.md"]) == 0.5            # rank 2 -> 1/2
    assert reciprocal_rank(ranked, ["a.md"]) == 1.0            # rank 1 -> 1/1
    assert reciprocal_rank(ranked, ["missing.md"]) == 0.0     # not retrieved


def test_abstention_counts_both_directions():
    answerable = [
        {"id": "q01", "abstained": False},
        {"id": "q02", "abstained": True},   # false abstention
    ]
    probes = [
        {"id": "q41", "abstained": True},   # correct
        {"id": "q42", "abstained": False},  # wrongly answered
    ]
    out = evaluate_abstention(answerable, probes)
    assert out["correct_abstentions"] == 1
    assert out["false_abstentions"] == 1
    assert out["abstention_recall"] == 0.5
    assert out["false_abstained_ids"] == ["q02"]
    assert out["wrongly_answered_ids"] == ["q42"]
