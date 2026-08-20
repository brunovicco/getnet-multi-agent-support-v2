"""Behavior tests for the semantic RAG retriever: ranking, thresholding, market isolation.

Uses a deterministic fake EmbeddingPort (no real network, no real Gemini quota) so cosine scores
are fully predictable, and a small controlled corpus fixture instead of the real one.
"""

import asyncio
import json
from pathlib import Path

from getnet_support.adapters.knowledge_retriever_semantic import SemanticKnowledgeRetriever
from getnet_support.domain.models import Market


class _KeywordEmbeddingPort:
    """Embeds each text as presence of two marker keywords: predictable, non-zero cosine scores."""

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        """Return a 2D one-hot-ish vector per text based on marker keyword presence."""
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append((1.0 if "alpha" in lowered else 0.0, 1.0 if "beta" in lowered else 0.0))
        return tuple(vectors)


def _write_corpus(corpus_dir: Path) -> None:
    common = {
        "language": "pt-BR",
        "topic": "test",
        "retrieved_at": "2026-08-19",
        "volatility": "low",
    }
    br_chunks = [
        {
            "id": "br-alpha",
            "title": "Alpha chunk",
            "text": "This chunk is about alpha.",
            "source": "https://example.test/alpha",
            "market": "BR",
            **common,
        },
        {
            "id": "br-beta",
            "title": "Beta chunk",
            "text": "This chunk is about beta.",
            "source": "https://example.test/beta",
            "market": "BR",
            **common,
        },
    ]
    global_chunks = [
        {
            "id": "global-alpha",
            "title": "Global alpha chunk",
            "text": "This chunk is also about alpha.",
            "source": "https://example.test/global-alpha",
            "market": "GLOBAL",
            **{**common, "language": "en"},
        },
    ]
    (corpus_dir / "getnet_br.json").write_text(json.dumps(br_chunks), encoding="utf-8")
    (corpus_dir / "getnet_global.json").write_text(json.dumps(global_chunks), encoding="utf-8")


def test_ranks_matching_chunk_above_threshold_and_filters_the_rest(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    retriever = asyncio.run(
        SemanticKnowledgeRetriever.create(_KeywordEmbeddingPort(), corpus_dir=tmp_path)
    )
    results = asyncio.run(retriever.retrieve("alpha", market=Market.BR, top_k=5))
    assert [item.chunk.id for item in results] == ["br-alpha"]
    assert results[0].score == 1.0


def test_market_isolation_holds_for_semantic_retrieval(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    retriever = asyncio.run(
        SemanticKnowledgeRetriever.create(_KeywordEmbeddingPort(), corpus_dir=tmp_path)
    )
    br_results = asyncio.run(retriever.retrieve("alpha", market=Market.BR, top_k=5))
    global_results = asyncio.run(retriever.retrieve("alpha", market=Market.GLOBAL, top_k=5))
    assert all(item.chunk.market is Market.BR for item in br_results)
    assert all(item.chunk.market is Market.GLOBAL for item in global_results)
    assert {item.chunk.id for item in global_results} == {"global-alpha"}
