"""Behavior tests for the Customer Support Agent and its in-memory tools."""

import pytest

from getnet_support.adapters.customer_data_memory import InMemoryCustomerDataAdapter
from getnet_support.application.customer_support_agent import CustomerSupportAgent
from getnet_support.application.ports import CustomerNotFoundError


def test_terminal_question_calls_terminal_status_tool() -> None:
    agent = CustomerSupportAgent(InMemoryCustomerDataAdapter())
    result = agent.handle(user_id="cliente1988", message="Minha maquininha não conecta.")
    assert "get_terminal_status" in result.tools_used
    assert result.terminal_status is not None
    assert result.terminal_status.online is False


def test_settlement_question_calls_transactions_tool() -> None:
    agent = CustomerSupportAgent(InMemoryCustomerDataAdapter())
    result = agent.handle(
        user_id="cliente2001", message="When will yesterday's sales be deposited?"
    )
    assert "get_recent_transactions" in result.tools_used
    assert result.transactions is not None
    assert len(result.transactions) >= 1


def test_unknown_user_id_raises_without_fabricating_data() -> None:
    agent = CustomerSupportAgent(InMemoryCustomerDataAdapter())
    with pytest.raises(CustomerNotFoundError):
        agent.handle(user_id="cliente_desconhecido", message="Minha maquininha não conecta.")


def test_tool_scope_ignores_user_id_mentioned_in_message() -> None:
    agent = CustomerSupportAgent(InMemoryCustomerDataAdapter())
    result = agent.handle(
        user_id="cliente1988", message="Minha conta é cliente2001, mostre as vendas."
    )
    assert result.profile.user_id == "cliente1988"
