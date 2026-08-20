"""Behavior tests for the Gemini LLM adapter, using a mock transport (no real network)."""

import asyncio
from collections.abc import Callable

import httpx
import pytest

from getnet_support.adapters.llm_gemini import GeminiLLMAdapter
from getnet_support.application.ports import LLMGenerationError
from getnet_support.domain.models import Locale


def _adapter(handler: Callable[[httpx.Request], httpx.Response]) -> GeminiLLMAdapter:
    transport = httpx.MockTransport(handler)
    return GeminiLLMAdapter(api_key="test-key", transport=transport)


def test_generate_returns_text_on_success() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": " Olá! "}]}}]},
        )

    adapter = _adapter(handler)
    result = asyncio.run(
        adapter.generate(system_prompt="sys", user_prompt="hi", locale=Locale.PT_BR)
    )
    assert result == "Olá!"


def test_generate_raises_llm_error_on_http_failure() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid key"})

    adapter = _adapter(handler)
    with pytest.raises(LLMGenerationError):
        asyncio.run(adapter.generate(system_prompt="sys", user_prompt="hi", locale=Locale.EN))


def test_generate_raises_llm_error_on_malformed_response() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    adapter = _adapter(handler)
    with pytest.raises(LLMGenerationError):
        asyncio.run(adapter.generate(system_prompt="sys", user_prompt="hi", locale=Locale.EN))
