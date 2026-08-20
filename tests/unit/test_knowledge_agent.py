"""Unit tests for the Knowledge Agent (REQ-09/10/11/12/13/15), fully offline."""

from getnet_support.application.agents.knowledge_agent import SENTINEL, KnowledgeAgent
from getnet_support.application.errors import LLMUnavailableError
from getnet_support.application.ports.web_search_port import WebSearchResult
from getnet_support.domain.chat import GroundingOrigin, Language, Market, Volatility
from getnet_support.domain.knowledge import CorpusChunk, RetrievedChunk


class _FakeRetriever:
    def __init__(self, candidates: list[RetrievedChunk]) -> None:
        self._candidates = candidates

    def search(
        self, query: str, *, market: Market | None = None, top_k: int = 3
    ) -> list[RetrievedChunk]:
        return self._candidates


class _FakeWebSearch:
    def __init__(self, results: list[WebSearchResult]) -> None:
        self._results = results

    def search(self, query: str, *, timeout_seconds: float) -> list[WebSearchResult]:
        return self._results


class _SentinelLLM:
    def generate(self, *, prompt: str, timeout_seconds: float) -> str:
        return SENTINEL


class _UnavailableLLM:
    def generate(self, *, prompt: str, timeout_seconds: float) -> str:
        raise LLMUnavailableError("down")


def _chunk(*, volatility: Volatility = Volatility.LOW) -> CorpusChunk:
    return CorpusChunk(
        id="c1",
        text="A Get Clássica custa R$ 79,90 por mês.",
        title="Get Clássica",
        source="https://site.getnet.com.br/maquininha/get-classica/",
        market=Market.BR,
        language=Language.PT_BR,
        topic="produtos",
        retrieved_at="2026-08-19",
        volatility=volatility,
    )


def test_accepted_evidence_never_calls_web() -> None:
    """REQ-11: KB-covered questions never pay for a web search call."""
    candidate = RetrievedChunk(chunk=_chunk(), score_retrieval=0.9, coverage=0.9)
    agent = KnowledgeAgent(
        retriever=_FakeRetriever([candidate]),
        llm=None,
        web_search=_FakeWebSearch([WebSearchResult("x", "https://x", "x", "2026-01-01")]),
        score_min=0.1,
        coverage_min=0.5,
        llm_timeout_seconds=2.0,
    )
    result = agent.answer("Quanto custa a Get Clássica?", market=None, language=Language.PT_BR)
    assert result.web_search_attempted is False
    assert result.grounding is GroundingOrigin.GETNET_KB


def test_rejected_evidence_attempts_web_and_degrades_without_key() -> None:
    """REQ-10/13: no accepted chunk -> web is attempted; without results, degrade honestly."""
    low_candidate = RetrievedChunk(chunk=_chunk(), score_retrieval=0.0, coverage=0.0)
    agent = KnowledgeAgent(
        retriever=_FakeRetriever([low_candidate]),
        llm=None,
        web_search=None,
        score_min=0.1,
        coverage_min=0.5,
        llm_timeout_seconds=2.0,
    )
    result = agent.answer("Quem foi Maradona?", market=None, language=Language.EN)
    assert result.web_search_attempted is True
    assert result.grounding is GroundingOrigin.NONE
    assert result.handoff_required is True
    assert SENTINEL not in result.answer


def test_sentinel_never_leaks_into_the_answer() -> None:
    """REQ-12: if the model emits the sentinel anyway, the raw context stands in."""
    candidate = RetrievedChunk(chunk=_chunk(), score_retrieval=0.9, coverage=0.9)
    agent = KnowledgeAgent(
        retriever=_FakeRetriever([candidate]),
        llm=_SentinelLLM(),
        web_search=None,
        score_min=0.1,
        coverage_min=0.5,
        llm_timeout_seconds=2.0,
    )
    result = agent.answer("Quanto custa a Get Clássica?", market=None, language=Language.PT_BR)
    assert SENTINEL not in result.answer


def test_llm_failure_falls_back_to_extractive_answer() -> None:
    """An unavailable LLM never turns into a fabricated or crashed response."""
    candidate = RetrievedChunk(chunk=_chunk(), score_retrieval=0.9, coverage=0.9)
    agent = KnowledgeAgent(
        retriever=_FakeRetriever([candidate]),
        llm=_UnavailableLLM(),
        web_search=None,
        score_min=0.1,
        coverage_min=0.5,
        llm_timeout_seconds=2.0,
    )
    result = agent.answer("Quanto custa a Get Clássica?", market=None, language=Language.PT_BR)
    assert result.answer == _chunk().text


def test_high_volatility_source_is_qualified() -> None:
    """REQ-15: high-volatility content is qualified, never presented as fixed."""
    candidate = RetrievedChunk(
        chunk=_chunk(volatility=Volatility.HIGH), score_retrieval=0.9, coverage=0.9
    )
    agent = KnowledgeAgent(
        retriever=_FakeRetriever([candidate]),
        llm=None,
        web_search=None,
        score_min=0.1,
        coverage_min=0.5,
        llm_timeout_seconds=2.0,
    )
    result = agent.answer("Quanto custa a Get Clássica?", market=None, language=Language.PT_BR)
    assert "confirme" in result.answer.lower()
