"""Behavior tests for the Gemini embedding adapter, using a mock transport (no real network)."""

import asyncio
from collections.abc import Callable

import httpx
import pytest

from getnet_support.adapters.embeddings_gemini import GeminiEmbeddingAdapter
from getnet_support.application.ports import EmbeddingGenerationError


def _adapter(handler: Callable[[httpx.Request], httpx.Response]) -> GeminiEmbeddingAdapter:
    return GeminiEmbeddingAdapter(api_key="test-key", transport=httpx.MockTransport(handler))


def test_embed_returns_one_vector_per_text_in_order() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}]},
        )

    adapter = _adapter(handler)
    result = asyncio.run(adapter.embed(("a", "b")))
    assert result == ((0.1, 0.2), (0.3, 0.4))


def test_embed_returns_empty_tuple_for_no_texts() -> None:
    adapter = GeminiEmbeddingAdapter(api_key="test-key")
    result = asyncio.run(adapter.embed(()))
    assert result == ()


def test_embed_raises_on_http_failure() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid key"})

    adapter = _adapter(handler)
    with pytest.raises(EmbeddingGenerationError):
        asyncio.run(adapter.embed(("a",)))


def test_embed_raises_on_malformed_response() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    adapter = _adapter(handler)
    with pytest.raises(EmbeddingGenerationError):
        asyncio.run(adapter.embed(("a",)))


def test_api_key_is_sent_as_header_never_in_the_url() -> None:
    """Regression test: httpx logs the full request URL at INFO level, so a `?key=` query
    parameter would leak the API key into every log stream. The key must only ever appear in a
    request header.
    """
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"embeddings": [{"values": [0.1]}]})

    adapter = _adapter(handler)
    asyncio.run(adapter.embed(("a",)))

    assert len(captured) == 1
    request = captured[0]
    assert "test-key" not in str(request.url)
    assert request.headers["x-goog-api-key"] == "test-key"
