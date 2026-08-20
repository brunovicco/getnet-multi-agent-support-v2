"""Unit tests for the real-embedding retriever (P1.1), with a fake SDK client."""

from dataclasses import dataclass
from typing import Any

import pytest

from getnet_support.adapters.retrieval.gemini_semantic_retriever import GeminiSemanticRetriever
from getnet_support.application.errors import LLMUnavailableError
from getnet_support.domain.chat import Language, Market, Volatility
from getnet_support.domain.knowledge import CorpusChunk


@dataclass(frozen=True, slots=True)
class _FakeEmbedding:
    values: list[float]


class _FakeEmbedContentResponse:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = [_FakeEmbedding(values=vector) for vector in embeddings]


class _FakeModels:
    """Returns a fixed vector per text, keyed by a simple substring match."""

    def __init__(self, vectors_by_keyword: dict[str, list[float]]) -> None:
        self._vectors_by_keyword = vectors_by_keyword
        self.calls: list[str] = []

    def embed_content(
        self, *, model: str, contents: list[str], config: object
    ) -> _FakeEmbedContentResponse:
        self.calls.append(model)
        vectors = []
        for text in contents:
            match = next((v for k, v in self._vectors_by_keyword.items() if k in text), [0.0, 0.0])
            vectors.append(match)
        return _FakeEmbedContentResponse(vectors)


class _FakeClient:
    def __init__(self, models: _FakeModels) -> None:
        self.models = models


def _chunk(chunk_id: str, title: str, text: str) -> CorpusChunk:
    return CorpusChunk(
        id=chunk_id,
        text=text,
        title=title,
        source=f"https://example.com/{chunk_id}",
        market=Market.BR,
        language=Language.PT_BR,
        topic="x",
        retrieved_at="2026-01-01",
        volatility=Volatility.LOW,
    )


def _build_retriever(monkeypatch: pytest.MonkeyPatch, models: _FakeModels) -> Any:
    monkeypatch.setattr(
        "getnet_support.adapters.retrieval.gemini_semantic_retriever.genai.Client",
        lambda **kwargs: _FakeClient(models),
    )
    corpus = (
        _chunk("classica", "Get Clássica", "maquininha de entrada"),
        _chunk("smart", "Get Smart", "maquininha inteligente"),
    )
    return GeminiSemanticRetriever("fake-key", corpus, timeout_seconds=2.0), corpus


def test_search_ranks_the_closer_embedding_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """The chunk whose embedding is closer to the query embedding ranks first."""
    models = _FakeModels(
        {
            "Clássica": [1.0, 0.0],
            "Smart": [0.0, 1.0],
            "query": [0.9, 0.1],
        }
    )
    retriever, _corpus = _build_retriever(monkeypatch, models)
    results = retriever.search("query about entrada", top_k=2)
    assert results[0].chunk.id == "classica"
    assert results[0].score_retrieval > results[1].score_retrieval


def test_query_embedding_failure_degrades_to_no_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-13: a failed query embedding never crashes the request; it yields no evidence."""
    models = _FakeModels({"Clássica": [1.0, 0.0], "Smart": [0.0, 1.0]})
    retriever, _ = _build_retriever(monkeypatch, models)

    def _raise(*args: object, **kwargs: object) -> None:
        raise LLMUnavailableError("down")

    monkeypatch.setattr(retriever, "_embed", _raise)
    assert retriever.search("anything") == []
