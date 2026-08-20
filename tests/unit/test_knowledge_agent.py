"""Behavior tests for the Knowledge Agent's RAG-first, web-search-fallback chain.

No keyword/regex classification decides whether to search the web: the local corpus is tried
first (authoritative for Getnet), and web search is the mandatory fallback whenever it doesn't
have enough evidence — regardless of what the question is about or how it's phrased. This
replaced an earlier keyword-based classifier that silently failed for phrasings it didn't
anticipate (e.g. "quantos graus vai fazer amanhã em São Paulo?", which didn't contain any of the
literal words "clima"/"previsão"/"temperatura").
"""

import asyncio

import pytest

from getnet_support.adapters.knowledge_retriever_local import LocalKnowledgeRetriever
from getnet_support.application.knowledge_agent import _NO_EVIDENCE_SENTINEL, KnowledgeAgent
from getnet_support.application.ports import LLMGenerationError
from getnet_support.domain.models import (
    KnowledgeChunk,
    Locale,
    Market,
    RetrievedChunk,
    WebSearchResult,
)


class _RaisingLLM:
    async def generate(self, *, system_prompt: str, user_prompt: str, locale: Locale) -> str:
        raise LLMGenerationError("no provider configured in tests")


class _ConfiguredWebSearch:
    """A healthy Tavily stand-in: always configured, always returns one result."""

    def is_configured(self) -> bool:
        return True

    async def search(self, query: str) -> tuple[WebSearchResult, ...]:
        return (WebSearchResult(title="Result", url="https://example.test", snippet="22C"),)


class _UnconfiguredWebSearch:
    def is_configured(self) -> bool:
        return False

    async def search(self, query: str) -> tuple[WebSearchResult, ...]:
        raise AssertionError("should not be called when unconfigured")


class _WebSearchThatMustNotBeCalled:
    """Fails the test if invoked; used to prove RAG-first short-circuits a good RAG match."""

    def is_configured(self) -> bool:
        raise AssertionError("web search must not be consulted when RAG already has evidence")

    async def search(self, query: str) -> tuple[WebSearchResult, ...]:
        raise AssertionError("web search must not be consulted when RAG already has evidence")


@pytest.mark.parametrize(
    "message",
    [
        "quantos graus vai fazer amanhã em São Paulo?",
        "vai chover amanhã em São Paulo?",
        "como está o tempo hoje?",
        "What's the weather forecast in Porto Alegre tomorrow?",
        "What's the euro exchange rate today?",
        "What is the capital of France?",
    ],
)
def test_questions_the_corpus_cannot_answer_fall_back_to_web_search(message: str) -> None:
    agent = KnowledgeAgent(
        retriever=LocalKnowledgeRetriever(), web_search=_ConfiguredWebSearch(), llm=_RaisingLLM()
    )
    result = asyncio.run(agent.handle(message=message, market=Market.BR, locale=Locale.PT_BR))
    assert result.tool_used == "tavily_web_search"
    assert result.sufficient_evidence is True


@pytest.mark.parametrize(
    "message",
    [
        "Qual a diferença entre Get Clássica e Get Smart?",
        "Quanto tempo demora a antecipação?",
    ],
)
def test_questions_the_corpus_can_answer_never_reach_web_search(message: str) -> None:
    agent = KnowledgeAgent(
        retriever=LocalKnowledgeRetriever(),
        web_search=_WebSearchThatMustNotBeCalled(),
        llm=_RaisingLLM(),
    )
    result = asyncio.run(agent.handle(message=message, market=Market.BR, locale=Locale.PT_BR))
    assert result.tool_used == "local_rag"
    assert result.sufficient_evidence is True


def test_escalates_with_no_fabrication_when_both_sources_lack_evidence() -> None:
    agent = KnowledgeAgent(
        retriever=LocalKnowledgeRetriever(),
        web_search=_UnconfiguredWebSearch(),
        llm=_RaisingLLM(),
    )
    result = asyncio.run(
        agent.handle(
            message="xyzzy unrelated gibberish nonsense", market=Market.BR, locale=Locale.EN
        )
    )
    assert result.sufficient_evidence is False
    assert result.sources == ()


class _FalsePositiveRetriever:
    """A retriever that scores above the evidence threshold for an irrelevant chunk.

    Simulates a real observed failure mode: semantic (embedding) similarity crossing the minimum
    score against a small corpus even for a completely unrelated proper-noun query (e.g. "quem
    foi Maradona?" scoring above threshold against payment-product chunks).
    """

    async def retrieve(
        self, query: str, *, market: Market, top_k: int = 3
    ) -> tuple[RetrievedChunk, ...]:
        chunk = KnowledgeChunk(
            id="br-irrelevant",
            text="Some unrelated Getnet product text.",
            title="Irrelevant Getnet chunk",
            source="https://example.test/irrelevant",
            market=market,
            language=Locale.PT_BR,
            topic="test",
            retrieved_at="2026-08-19",
            volatility="low",
        )
        return (RetrievedChunk(chunk=chunk, score=1.0),)


class _LLMThatDetectsIrrelevantContext:
    """Simulates Gemini correctly recognizing retrieved context doesn't answer the question."""

    async def generate(self, *, system_prompt: str, user_prompt: str, locale: Locale) -> str:
        if "Irrelevant Getnet chunk" in user_prompt:
            return _NO_EVIDENCE_SENTINEL
        return "Diego Maradona was an Argentine football legend."


def test_high_retrieval_score_alone_does_not_override_the_llms_own_judgment() -> None:
    """Regression test for a real production bug: a chunk that clears the retrieval-score
    threshold but does not actually answer the question must not be presented as a resolved
    answer — the LLM's explicit "I don't have this" signal must still trigger the web fallback.
    """
    fake_web_result = (
        WebSearchResult(
            title="Diego Maradona",
            url="https://example.test/maradona",
            snippet="Argentine footballer",
        ),
    )

    class _WebSearchWithRealAnswer:
        def is_configured(self) -> bool:
            return True

        async def search(self, query: str) -> tuple[WebSearchResult, ...]:
            return fake_web_result

    agent = KnowledgeAgent(
        retriever=_FalsePositiveRetriever(),
        web_search=_WebSearchWithRealAnswer(),
        llm=_LLMThatDetectsIrrelevantContext(),
    )
    result = asyncio.run(
        agent.handle(message="quem foi Maradona?", market=Market.BR, locale=Locale.PT_BR)
    )
    assert result.tool_used == "tavily_web_search"
    assert result.sufficient_evidence is True
    assert "Maradona" in result.answer
