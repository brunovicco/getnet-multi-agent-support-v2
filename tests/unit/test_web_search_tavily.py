"""Behavior tests for the Tavily web search adapter, using a mock transport (no real network)."""

import asyncio
from collections.abc import Callable

import httpx
import pytest

from getnet_support.adapters.web_search_tavily import TavilyWebSearchAdapter
from getnet_support.application.ports import WebSearchUnavailableError


def _adapter(
    api_key: str | None, handler: Callable[[httpx.Request], httpx.Response] | None = None
) -> TavilyWebSearchAdapter:
    transport = httpx.MockTransport(handler) if handler is not None else None
    return TavilyWebSearchAdapter(api_key=api_key, transport=transport)


def test_is_configured_reflects_api_key_presence() -> None:
    assert _adapter("a-key").is_configured() is True
    assert _adapter(None).is_configured() is False


def test_search_raises_unavailable_without_api_key() -> None:
    adapter = _adapter(None)
    with pytest.raises(WebSearchUnavailableError):
        asyncio.run(adapter.search("weather in Porto Alegre tomorrow"))


def test_search_returns_results_on_success() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Weather", "url": "https://example.test/w", "content": "Sunny"}
                ]
            },
        )

    adapter = _adapter("key", handler)
    results = asyncio.run(adapter.search("weather in Porto Alegre tomorrow"))
    assert len(results) == 1
    assert results[0].title == "Weather"


def test_search_raises_unavailable_on_http_failure() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    adapter = _adapter("key", handler)
    with pytest.raises(WebSearchUnavailableError):
        asyncio.run(adapter.search("euro exchange rate today"))
