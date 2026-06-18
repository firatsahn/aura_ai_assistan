"""Sparse encoder — turns text into BM25-style sparse term vectors.

The lexical counterpart of the dense `embedding` module: indexing and query
time both go through here so documents and queries hash tokens identically and
their sparse vectors share an index space (the precondition for any match).
"""

from .encoder import SparseVector, encode_documents, encode_query

__all__ = ["SparseVector", "encode_documents", "encode_query"]
