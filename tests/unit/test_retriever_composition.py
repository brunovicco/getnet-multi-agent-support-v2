"""Unit tests for retriever selection and fallback (P1.1), fully offline."""

import pytest

from getnet_support.adapters.retrieval.corpus_loader import load_corpus
from getnet_support.adapters.retrieval.lexical_retriever import LexicalRetriever
from getnet_support.adapters.retrieval.semantic_retriever import SemanticRetriever
from getnet_support.application.errors import LLMUnavailableError
from getnet_support.entrypoints.http import _build_retriever
from getnet_support.entrypoints.settings import Settings


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_lexical_is_the_default() -> None:
    """The default configuration builds the lexical retriever."""
    retriever, mode = _build_retriever(_settings(), load_corpus())
    assert isinstance(retriever, LexicalRetriever)
    assert mode == "lexical"


def test_local_semantic_mode_builds_without_any_key() -> None:
    """`RETRIEVER=semantic` never needs a key (T02/REQ-16)."""
    retriever, mode = _build_retriever(_settings(retriever="semantic"), load_corpus())
    assert isinstance(retriever, SemanticRetriever)
    assert mode == "semantic"


def test_semantic_embeddings_without_key_falls_back_to_local_semantic() -> None:
    """REQ-24: no `GOOGLE_API_KEY` -> degrade explicitly, never crash."""
    retriever, mode = _build_retriever(
        _settings(retriever="semantic_embeddings", google_api_key=""), load_corpus()
    )
    assert isinstance(retriever, SemanticRetriever)
    assert mode == "semantic"


def test_semantic_embeddings_falls_back_when_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed embedding call at startup degrades instead of crashing the process."""

    class _FailingRetriever:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise LLMUnavailableError("no network in this test")

    monkeypatch.setattr(
        "getnet_support.entrypoints.http.GeminiSemanticRetriever", _FailingRetriever
    )
    retriever, mode = _build_retriever(
        _settings(retriever="semantic_embeddings", google_api_key="fake-key"), load_corpus()
    )
    assert isinstance(retriever, SemanticRetriever)
    assert mode == "semantic"


def test_semantic_embeddings_uses_the_real_retriever_when_construction_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful embedding call at startup uses the real retriever."""

    class _FakeRetriever:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr("getnet_support.entrypoints.http.GeminiSemanticRetriever", _FakeRetriever)
    retriever, mode = _build_retriever(
        _settings(retriever="semantic_embeddings", google_api_key="fake-key"), load_corpus()
    )
    assert isinstance(retriever, _FakeRetriever)
    assert mode == "semantic_embeddings"
