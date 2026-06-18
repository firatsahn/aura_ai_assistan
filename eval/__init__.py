"""Evaluation harness: golden-set metrics for retrieval, generation, abstention.

A top-level package (sibling of `backend`) so the harness runs as
`python -m eval.run` from the project root and imports the real RAG pieces
(`embed`, `search`, `answer_question`) instead of re-implementing them.
"""
