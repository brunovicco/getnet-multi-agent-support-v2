"""Unit tests for Support -> Knowledge chaining (P1.3), fully offline.

Wires the real agents together (no HTTP, no mocks) — the same components
`entrypoints/http.py` composes, backed by the real committed corpus and the
in-memory customer fixtures, exactly like `tests/unit/test_retrieval.py`
already does for the retriever alone.
"""

from getnet_support.adapters.customer.in_memory_customer_repository import (
    InMemoryCustomerRepository,
)
from getnet_support.adapters.retrieval.corpus_loader import load_corpus
from getnet_support.adapters.retrieval.lexical_retriever import LexicalRetriever
from getnet_support.application.agents.customer_support_agent import CustomerSupportAgent
from getnet_support.application.agents.escalation_agent import EscalationAgent
from getnet_support.application.agents.knowledge_agent import KnowledgeAgent
from getnet_support.application.agents.router_agent import RouterAgent
from getnet_support.application.chat_service import ChatApplicationService


def _chat_service() -> ChatApplicationService:
    knowledge_agent = KnowledgeAgent(
        retriever=LexicalRetriever(load_corpus()),
        llm=None,
        web_search=None,
        score_min=0.1,
        coverage_min=0.55,
        llm_timeout_seconds=2.0,
    )
    return ChatApplicationService(
        router=RouterAgent(),
        knowledge_agent=knowledge_agent,
        customer_support_agent=CustomerSupportAgent(InMemoryCustomerRepository()),
        escalation_agent=EscalationAgent(),
    )


def test_disconnected_terminal_chains_in_the_matching_kb_article() -> None:
    """P1.3: a real terminal problem pulls the troubleshooting KB article in."""
    result = _chat_service().handle(
        message="My card machine won't connect to the internet, what should I do?",
        user_id="cliente1988",
    )
    assert result.agents == ["router", "customer_support", "knowledge"]
    assert result.grounding.value == "customer_data"
    assert any(source.origin.value == "getnet_kb" for source in result.sources)
    assert "This might also help:" in result.answer


def test_healthy_terminal_does_not_chain() -> None:
    """No real problem to troubleshoot -> no chaining, no extra KB source."""
    result = _chat_service().handle(
        message="My card machine won't connect to the internet, what should I do?",
        user_id="cliente2001",
    )
    assert result.agents == ["router", "customer_support"]
    assert result.sources == []


def test_transaction_intent_never_chains() -> None:
    """Chaining is scoped to the terminal-troubleshooting case, not every support turn."""
    result = _chat_service().handle(
        message="Quando cai o dinheiro da venda de ontem?", user_id="cliente1988"
    )
    assert result.agents == ["router", "customer_support"]
    assert result.sources == []
