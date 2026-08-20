"""Behavior tests for the Knowledge Agent's web-search-vs-RAG strategy selection.

Regression coverage for a real gap found in production use: PT-BR weather phrasings without the
literal words "clima"/"previsão"/"temperatura" (e.g. "quantos graus vai fazer amanhã em São
Paulo?") were silently falling through to RAG instead of Tavily.
"""

import asyncio

import pytest

from getnet_support.adapters.knowledge_retriever_local import LocalKnowledgeRetriever
from getnet_support.application.knowledge_agent import KnowledgeAgent
from getnet_support.application.ports import LLMGenerationError
from getnet_support.domain.models import Locale, Market, WebSearchResult


class _RaisingLLM:
    async def generate(self, *, system_prompt: str, user_prompt: str, locale: Locale) -> str:
        raise LLMGenerationError("no provider configured in tests")


class _ConfiguredWebSearch:
    def is_configured(self) -> bool:
        return True

    async def search(self, query: str) -> tuple[WebSearchResult, ...]:
        return (WebSearchResult(title="Forecast", url="https://example.test", snippet="22C"),)


def _agent() -> KnowledgeAgent:
    return KnowledgeAgent(
        retriever=LocalKnowledgeRetriever(),
        web_search=_ConfiguredWebSearch(),
        llm=_RaisingLLM(),
    )


@pytest.mark.parametrize(
    "message",
    [
        "quantos graus vai fazer amanhã em São Paulo?",
        "vai chover amanhã em São Paulo?",
        "como está o tempo hoje?",
        "What's the weather forecast in Porto Alegre tomorrow?",
        "What's the euro exchange rate today?",
    ],
)
def test_current_info_phrasings_use_web_search(message: str) -> None:
    result = asyncio.run(_agent().handle(message=message, market=Market.BR, locale=Locale.PT_BR))
    assert result.tool_used == "tavily_web_search"


@pytest.mark.parametrize(
    "message",
    [
        "Qual a diferença entre Get Clássica e Get Smart?",
        "Quanto tempo demora a antecipação?",
    ],
)
def test_product_questions_use_rag_not_web_search(message: str) -> None:
    result = asyncio.run(_agent().handle(message=message, market=Market.BR, locale=Locale.PT_BR))
    assert result.tool_used == "local_rag"
