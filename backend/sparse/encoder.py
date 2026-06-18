"""Sparse (BM25-style) text encoder: text -> sparse term vectors.

The dense embedding captures meaning; this captures *literal tokens*. Exact
strings that must match verbatim — error codes (``E-102``), specs (``5V/2A``),
protocol versions (``Zigbee 3.0``) — are exactly where dense similarity is weak
and lexical overlap wins. Step 3b fuses both rankings (see vectorstore.search).

We deliberately roll a tiny tokenizer instead of pulling in fastembed: no extra
dependency, no model download, fully hermetic tests, and full control over
tokenization so the exact-match tokens above survive intact.

How the BM25 split works with Qdrant:
- Document side (``encode_documents``) stores the BM25 term-frequency saturation
  component, length-normalized against the corpus average document length.
- Query side (``encode_query``) stores raw term counts, no length normalization.
- The IDF factor is **not** stored here — Qdrant applies it server-side via the
  sparse vector's ``Modifier.IDF``, so IDF reflects the whole collection and we
  never have to recompute it when the corpus changes.

Token -> index uses a stable hash (``blake2b``), never Python's builtin
``hash()`` for strings: that is salted per process (``PYTHONHASHSEED``), which
would make a document indexed in one process and a query encoded in another map
the same token to different indices — silently breaking every lexical match.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

# Keep intra-token connectors (-, /, .) between alphanumerics so "E-102",
# "5v/2a" and "3.0" stay single tokens instead of being split apart. \w is
# unicode by default in Python 3, so Turkish letters (ç, ğ, ı, ö, ş, ü) match.
_TOKEN_RE = re.compile(r"\w+(?:[-/.]\w+)*", re.UNICODE)

# Combining dot above: casefold("İ") -> "i" + this mark; stripped in _tokenize.
_COMBINING_DOT = "\u0307"

# Token index space. Large enough that collisions across the corpus's few
# thousand distinct tokens are negligible.
_VOCAB_SIZE = 2**31

# BM25 parameters. k1 controls term-frequency saturation; b controls how much
# document length normalizes the score. These are the standard defaults.
_K1 = 1.5
_B = 0.75


@dataclass
class SparseVector:
    """A sparse vector as parallel (token index, weight) arrays."""

    indices: list[int]
    values: list[float]


def _tokenize(text: str) -> list[str]:
    """Lowercase (Turkish-aware) and split into match-preserving tokens.

    `casefold()` maps the Turkish dotted capital "İ" to "i" + U+0307 (combining
    dot above); that combining mark is not a word character, so it would split
    the token (and stop "İnternet" matching "internet"). Strip it so the dotted
    and dotless forms collapse to the same token.
    """
    return _TOKEN_RE.findall(text.casefold().replace(_COMBINING_DOT, ""))


def _hash(token: str) -> int:
    """Map a token to a stable index, identical across processes and machines."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _VOCAB_SIZE


def encode_documents(texts: list[str]) -> list[SparseVector]:
    """Encode documents with BM25 tf-saturation, length-normalized by avgdl.

    ``avgdl`` is the mean token count over ``texts``. At index time the whole
    corpus is passed in one call, so this is the true corpus average.
    """
    if not texts:
        return []

    tokenized = [_tokenize(t) for t in texts]
    lengths = [len(toks) for toks in tokenized]
    avgdl = (sum(lengths) / len(lengths)) or 1.0  # guard all-empty corpus

    vectors: list[SparseVector] = []
    for toks, dl in zip(tokenized, lengths):
        counts = Counter(_hash(tok) for tok in toks)
        norm = _K1 * (1.0 - _B + _B * dl / avgdl)
        indices: list[int] = []
        values: list[float] = []
        for index, tf in counts.items():
            indices.append(index)
            # BM25 tf component: saturates as tf grows, bounded by k1 + 1.
            values.append(tf * (_K1 + 1.0) / (tf + norm))
        vectors.append(SparseVector(indices=indices, values=values))
    return vectors


def encode_query(text: str) -> SparseVector:
    """Encode a query as raw term counts (IDF is applied by Qdrant)."""
    counts = Counter(_hash(tok) for tok in _tokenize(text))
    return SparseVector(
        indices=list(counts.keys()),
        values=[float(tf) for tf in counts.values()],
    )
