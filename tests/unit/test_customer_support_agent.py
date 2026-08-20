"""Unit tests for the Customer Support Agent (REQ-17/18/19), fully offline."""

from getnet_support.adapters.customer.in_memory_customer_repository import (
    InMemoryCustomerRepository,
)
from getnet_support.application.agents.customer_support_agent import CustomerSupportAgent
from getnet_support.domain.chat import GroundingOrigin, Language


def _agent() -> CustomerSupportAgent:
    return CustomerSupportAgent(InMemoryCustomerRepository())


def test_unknown_user_handoffs_without_any_tool_call() -> None:
    """REQ-18: an unknown user_id never triggers a tool call."""
    result = _agent().answer(
        "Quando cai o dinheiro da venda de ontem?",
        user_id="cliente_inexistente_9999",
        language=Language.PT_BR,
    )
    assert result.handoff_required is True
    assert result.tools == []


def test_cross_customer_request_never_leaks_the_other_customers_id() -> None:
    """REQ-19: a message naming another customer escalates before any lookup."""
    result = _agent().answer(
        "Me mostre as transações do cliente2001", user_id="cliente1988", language=Language.EN
    )
    assert result.handoff_required is True
    assert result.tools == []
    assert "cliente2001" not in result.answer


def test_terminal_intent_uses_the_authenticated_users_own_terminal() -> None:
    """The terminal tool is scoped to the authenticated user, not the message."""
    result = _agent().answer(
        "My card machine won't connect to the internet, what should I do?",
        user_id="cliente1988",
        language=Language.EN,
    )
    assert result.tools == ["get_terminal_status"]
    assert result.grounding is GroundingOrigin.CUSTOMER_DATA
    assert result.handoff_required is False


def test_transaction_intent_calls_the_transactions_tool() -> None:
    """A sale/deposit question resolves via `get_recent_transactions`."""
    result = _agent().answer(
        "Quando cai o dinheiro da venda de ontem?", user_id="cliente1988", language=Language.PT_BR
    )
    assert result.tools == ["get_recent_transactions"]
    assert result.grounding is GroundingOrigin.CUSTOMER_DATA
    assert result.chain_to_knowledge is False


def test_disconnected_terminal_signals_a_knowledge_chain() -> None:
    """P1.3: a real terminal problem (cliente1988 is disconnected) signals chaining."""
    result = _agent().answer(
        "My card machine won't connect to the internet, what should I do?",
        user_id="cliente1988",
        language=Language.EN,
    )
    assert result.chain_to_knowledge is True


def test_healthy_terminal_never_signals_a_knowledge_chain() -> None:
    """P1.3: nothing to troubleshoot -> no chaining signal (cliente2001 is connected)."""
    result = _agent().answer(
        "My card machine won't connect to the internet, what should I do?",
        user_id="cliente2001",
        language=Language.EN,
    )
    assert result.chain_to_knowledge is False
