"""Acceptance tests for the wired chat API: Router -> agents -> guardrails -> response.

Uses fake LLM/web-search ports (no real network); the retriever, customer tools, router, and
guardrails are exercised for real.
"""

import pytest
from fastapi.testclient import TestClient

from getnet_support.adapters.customer_data_memory import InMemoryCustomerDataAdapter
from getnet_support.adapters.knowledge_retriever_local import LocalKnowledgeRetriever
from getnet_support.application.chat_service import ChatApplicationService
from getnet_support.application.customer_support_agent import CustomerSupportAgent
from getnet_support.application.escalation_agent import EscalationAgent
from getnet_support.application.knowledge_agent import KnowledgeAgent
from getnet_support.application.ports import LLMGenerationError, WebSearchPort
from getnet_support.application.router_agent import RouterAgent
from getnet_support.domain.models import Locale, WebSearchResult
from getnet_support.entrypoints.http import create_app


class _RaisingLLM:
    """Always fails, forcing the deterministic extractive fallback in every test."""

    async def generate(self, *, system_prompt: str, user_prompt: str, locale: Locale) -> str:
        """Simulate no LLM provider being configured."""
        raise LLMGenerationError("no provider configured in tests")


class _UnconfiguredWebSearch:
    """Reports as unconfigured, like a deployment with no TAVILY_API_KEY."""

    def is_configured(self) -> bool:
        """Report that no provider credentials are set."""
        return False

    async def search(self, query: str) -> tuple[WebSearchResult, ...]:
        """Fail the test if called; the Knowledge Agent must check is_configured() first."""
        raise AssertionError("should not be called when unconfigured")


class _FakeWebSearch:
    """Returns a fixed set of results, like a successful Tavily call."""

    def __init__(self, results: tuple[WebSearchResult, ...]) -> None:
        """Bind the canned results returned by every search call."""
        self._results = results

    def is_configured(self) -> bool:
        """Report as configured."""
        return True

    async def search(self, query: str) -> tuple[WebSearchResult, ...]:
        """Return the canned results regardless of the query."""
        return self._results


def _build_service(web_search: WebSearchPort | None = None) -> ChatApplicationService:
    return ChatApplicationService(
        router=RouterAgent(),
        knowledge_agent=KnowledgeAgent(
            retriever=LocalKnowledgeRetriever(),
            web_search=web_search or _UnconfiguredWebSearch(),
            llm=_RaisingLLM(),
        ),
        customer_support_agent=CustomerSupportAgent(InMemoryCustomerDataAdapter()),
        escalation_agent=EscalationAgent(),
    )


@pytest.fixture
def client() -> TestClient:
    """A TestClient wired with fake LLM/web-search ports and real everything else."""
    return TestClient(create_app(chat_service=_build_service()))


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_serves_ui(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@pytest.mark.contract
def test_chat_endpoint_returns_full_contract(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={
            "message": "Qual a diferença entre Get Clássica e Get Smart?",
            "user_id": "cliente1988",
        },
    )
    assert response.status_code == 200
    body = response.json()
    expected_fields = (
        "answer",
        "sources",
        "route",
        "agents",
        "tools",
        "handoff_required",
        "trace_id",
        "latency_ms",
    )
    for field in expected_fields:
        assert field in body


def test_product_question_routes_to_knowledge_with_source(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={
            "message": "Qual a diferença entre Get Clássica e Get Smart?",
            "user_id": "cliente1988",
        },
    )
    body = response.json()
    assert body["route"] == "knowledge"
    assert body["sources"]
    assert body["sources"][0]["market"] == "BR"


def test_web_question_uses_configured_tavily_adapter() -> None:
    fake_results = (
        WebSearchResult(title="Weather PoA", url="https://x.test", snippet="Sunny, 22C"),
    )
    service = _build_service(web_search=_FakeWebSearch(fake_results))
    client = TestClient(create_app(chat_service=service))
    response = client.post(
        "/chat",
        json={
            "message": "What's the weather forecast in Porto Alegre tomorrow?",
            "user_id": "cliente1988",
        },
    )
    body = response.json()
    assert body["tools"] == ["tavily_web_search"]
    assert body["handoff_required"] is False


def test_web_search_unavailable_returns_safe_fallback(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"message": "What's the euro exchange rate today?", "user_id": "cliente1988"},
    )
    body = response.json()
    assert body["handoff_required"] is True
    assert body["sources"] == []
    assert "unavailable" in body["answer"].lower() or "indispon" in body["answer"].lower()


def test_settlement_question_uses_transaction_tool(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={
            "message": "When will the money from yesterday's sales be deposited?",
            "user_id": "cliente2001",
        },
    )
    body = response.json()
    assert "get_recent_transactions" in body["tools"]
    assert body["route"] == "customer_support"


def test_terminal_question_chains_support_and_knowledge(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"message": "Minha maquininha não conecta à internet.", "user_id": "cliente1988"},
    )
    body = response.json()
    assert "get_terminal_status" in body["tools"]
    assert "knowledge" in body["agents"]
    assert body["sources"]


def test_unknown_user_triggers_escalation_without_fabricated_data(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"message": "Minha maquininha não conecta.", "user_id": "cliente_desconhecido"},
    )
    body = response.json()
    assert body["handoff_required"] is True
    assert body["route"] == "customer_support"
    assert "escalation" in body["agents"]


def test_chat_responds_in_portuguese_when_locale_is_pt_br(client: TestClient) -> None:
    response = client.post(
        "/chat", json={"message": "Pix", "user_id": "cliente1988", "locale": "pt-BR"}
    )
    body = response.json()
    assert "fonte oficial" in body["answer"].lower()


def test_chat_responds_in_english_when_locale_is_en(client: TestClient) -> None:
    response = client.post(
        "/chat", json={"message": "Pix", "user_id": "cliente1988", "locale": "en"}
    )
    body = response.json()
    assert "official source" in body["answer"].lower()


def test_br_market_never_returns_global_sources(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={
            "message": "What products does Getnet offer?",
            "user_id": "cliente1988",
            "market": "BR",
        },
    )
    body = response.json()
    assert all(source["market"] == "BR" for source in body["sources"])


def test_global_market_never_returns_br_sources(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={
            "message": "What products does Getnet offer?",
            "user_id": "cliente1988",
            "market": "GLOBAL",
        },
    )
    body = response.json()
    assert all(source["market"] == "GLOBAL" for source in body["sources"])


def test_unsupported_financial_operation_escalates(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"message": "I want a refund and to dispute a charge.", "user_id": "cliente1988"},
    )
    body = response.json()
    assert body["route"] == "escalation"
    assert body["handoff_required"] is True
