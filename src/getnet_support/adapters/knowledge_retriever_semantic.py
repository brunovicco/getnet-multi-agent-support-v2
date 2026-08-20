"""Semantic RAG retriever: cosine similarity over Gemini embeddings (P1 upgrade).

Same persisted corpus and chunk metadata as `knowledge_retriever_local.py` — only the scoring
strategy changes: `top_k = 3-5`, similarity, and provenance. A match below `_MIN_COSINE_SCORE` is
filtered out entirely rather than trusting the shared minimum-evidence guardrail to catch it,
because cosine similarity and the lexical retriever's IDF score live on different numeric scales.
"""

import math
from pathlib import Path
from typing import Self

from getnet_support.adapters.corpus_loader import load_corpus_chunks
from getnet_support.application.ports import EmbeddingPort, KnowledgeRetrieverPort
from getnet_support.domain.models import KnowledgeChunk, Market, RetrievedChunk

_MIN_COSINE_SCORE = 0.3


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(x * x for x in left))
    norm_right = math.sqrt(sum(y * y for y in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


class SemanticKnowledgeRetriever(KnowledgeRetrieverPort):
    """Cosine-similarity retriever over Gemini embeddings of the persisted corpus."""

    def __init__(
        self,
        chunks: tuple[KnowledgeChunk, ...],
        chunk_embeddings: dict[str, tuple[float, ...]],
        embedding_port: EmbeddingPort,
    ) -> None:
        """Bind precomputed corpus embeddings; prefer `create` over calling this directly."""
        self._chunks = chunks
        self._chunk_embeddings = chunk_embeddings
        self._embedding_port = embedding_port

    @classmethod
    async def create(cls, embedding_port: EmbeddingPort, corpus_dir: Path | None = None) -> Self:
        """Load the persisted corpus and embed every chunk once, up front.

        Raises EmbeddingGenerationError (propagated from the port) if the provider call fails;
        the caller decides whether to fall back to lexical retrieval.
        """
        chunks = load_corpus_chunks(corpus_dir)
        vectors = await embedding_port.embed(tuple(f"{c.title} {c.text}" for c in chunks))
        chunk_embeddings = dict(zip((c.id for c in chunks), vectors, strict=True))
        return cls(chunks, chunk_embeddings, embedding_port)

    async def retrieve(
        self, query: str, *, market: Market, top_k: int = 3
    ) -> tuple[RetrievedChunk, ...]:
        """Return the top matching chunks scoped to market, best cosine score first."""
        (query_vector,) = await self._embedding_port.embed((query,))
        scored = [
            RetrievedChunk(
                chunk=chunk,
                score=_cosine_similarity(query_vector, self._chunk_embeddings[chunk.id]),
            )
            for chunk in self._chunks
            if chunk.market is market
        ]
        relevant = [item for item in scored if item.score >= _MIN_COSINE_SCORE]
        relevant.sort(key=lambda item: item.score, reverse=True)
        return tuple(relevant[:top_k])
