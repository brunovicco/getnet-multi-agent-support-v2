"""Deterministic bag-of-words cosine retriever over the committed corpus."""

from collections import Counter
from math import sqrt

from getnet_support.domain.chat import Market
from getnet_support.domain.knowledge import (
    CorpusChunk,
    RetrievedChunk,
    content_terms,
    coverage_lexical,
)


def _cosine(query_vector: Counter[str], chunk_vector: Counter[str]) -> float:
    """Cosine similarity between two term-count vectors."""
    if not query_vector or not chunk_vector:
        return 0.0
    dot = sum(count * chunk_vector.get(term, 0) for term, count in query_vector.items())
    norm_query = sqrt(sum(count * count for count in query_vector.values()))
    norm_chunk = sqrt(sum(count * count for count in chunk_vector.values()))
    if norm_query == 0 or norm_chunk == 0:
        return 0.0
    return dot / (norm_query * norm_chunk)


class LexicalRetriever:
    """`RetrieverPort` implementation using literal/canonicalized token overlap."""

    def __init__(self, corpus: tuple[CorpusChunk, ...]) -> None:
        """Precompute one term-count vector per corpus chunk."""
        self._corpus = corpus
        self._vectors = {
            chunk.id: Counter(content_terms(f"{chunk.title} {chunk.text}")) for chunk in corpus
        }

    def search(
        self, query: str, *, market: Market | None = None, top_k: int = 3
    ) -> list[RetrievedChunk]:
        """Rank corpus chunks by cosine similarity, restricted to `market` if given."""
        query_vector = Counter(content_terms(query))
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
