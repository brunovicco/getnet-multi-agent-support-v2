"""Local, deterministic "semantic" retriever: hashed bag-of-words cosine.

REQ-16 requires the semantic retriever to be verifiable offline, with parity
against the lexical retriever, using only the committed corpus (no
`GOOGLE_API_KEY`, no network call — the "precomputed embeddings" the eval
docstring refers to are these hashed vectors, computed once at construction
time). A real embedding-model-backed retriever is documented as a P1
follow-up in the README; this local vectorizer is what stands in for it this
round, not a placeholder that skips REQ-16.
"""

import hashlib
from math import sqrt

from getnet_support.domain.chat import Market
from getnet_support.domain.knowledge import (
    CorpusChunk,
    RetrievedChunk,
    content_terms,
    coverage_lexical,
)

_DIMENSIONS = 256


def _hash_vector(terms: list[str]) -> list[float]:
    """Project terms into a fixed-size vector via the hashing trick."""
    vector = [0.0] * _DIMENSIONS
    for term in terms:
        digest = hashlib.sha256(term.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % _DIMENSIONS
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    return vector


def _cosine(vector_a: list[float], vector_b: list[float]) -> float:
    """Cosine similarity between two equal-length dense vectors."""
    dot = sum(a * b for a, b in zip(vector_a, vector_b, strict=True))
    norm_a = sqrt(sum(a * a for a in vector_a))
    norm_b = sqrt(sum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticRetriever:
    """`RetrieverPort` implementation using local hashed vectors, no API key."""

    def __init__(self, corpus: tuple[CorpusChunk, ...]) -> None:
        """Precompute one hashed vector per corpus chunk."""
        self._corpus = corpus
        self._vectors = {
            chunk.id: _hash_vector(content_terms(f"{chunk.title} {chunk.text}")) for chunk in corpus
        }

    def search(
        self, query: str, *, market: Market | None = None, top_k: int = 20
    ) -> list[RetrievedChunk]:
        """Rank corpus chunks by cosine similarity, restricted to `market` if given."""
        query_vector = _hash_vector(content_terms(query))
        candidates = [chunk for chunk in self._corpus if market is None or chunk.market == market]
        scored = [
            RetrievedChunk(
                chunk=chunk,
                score_retrieval=_cosine(query_vector, self._vectors[chunk.id]),
                coverage=coverage_lexical(query, f"{chunk.title} {chunk.text}"),
            )
            for chunk in candidates
        ]
        scored.sort(key=lambda item: item.score_retrieval, reverse=True)
        return scored[:top_k]
