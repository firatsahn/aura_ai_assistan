"""Sparse BM25 encoder: tokenization, stable hashing, BM25 shape. Hermetic."""

from backend.sparse import SparseVector, encode_documents, encode_query
from backend.sparse.encoder import _K1, _hash, _tokenize


def test_exact_match_tokens_survive():
    # The connectors -, /, . must stay *inside* the token, not split it.
    assert "e-102" in _tokenize("Hata E-102 oluştu")
    assert "5v/2a" in _tokenize("Çıkış 5V/2A olmalı")
    assert set(_tokenize("Zigbee 3.0 destekler")) >= {"zigbee", "3.0"}


def test_turkish_casefold_is_consistent():
    # Upper- and lower-case forms must reduce to the same token (so the same
    # index), or a lexical match would depend on letter case.
    assert _tokenize("İNTERNET") == _tokenize("internet")
    assert _hash("internet") == _hash(_tokenize("İNTERNET")[0])


def test_hash_is_stable_not_salted():
    # A fixed token must map to a fixed index regardless of process / run. If
    # someone swapped in the builtin (salted) hash(), this asserted value breaks.
    assert _hash("e-102") == _hash("e-102")
    expected = _hash("zigbee")
    assert isinstance(expected, int) and 0 <= expected < 2**31
    # Pin the value so an accidental switch to a salted/changed hash is caught.
    assert _hash("e-102") == 280454800


def test_document_and_query_share_index_space():
    # A token shared by a document and a query must land on the same index, or
    # the two sparse vectors can never match in Qdrant.
    [doc] = encode_documents(["cihaz hata E-102 verdi"])
    query = encode_query("E-102 nedir")
    shared = _hash("e-102")
    assert shared in doc.indices
    assert shared in query.indices


def test_bm25_values_positive_and_saturated():
    # Document side: values > 0 and bounded by the tf-saturation ceiling k1+1.
    [doc] = encode_documents(["alfa beta beta beta gama"])
    assert doc.indices and len(doc.indices) == len(doc.values)
    assert all(v > 0 for v in doc.values)
    assert all(v < _K1 + 1.0 for v in doc.values)
    # The thrice-repeated term outweighs a once-seen term, but not 3x (saturation).
    beta = doc.values[doc.indices.index(_hash("beta"))]
    alfa = doc.values[doc.indices.index(_hash("alfa"))]
    assert alfa < beta < 3 * alfa


def test_query_values_are_raw_counts():
    q = encode_query("hata hata kodu")
    assert q.values[q.indices.index(_hash("hata"))] == 2.0
    assert q.values[q.indices.index(_hash("kodu"))] == 1.0


def test_empty_inputs():
    assert encode_documents([]) == []
    assert encode_query("") == SparseVector(indices=[], values=[])
    assert encode_query("   .,;  ") == SparseVector(indices=[], values=[])
