"""Behavior tests for the reason-aware Escalation Agent."""

from getnet_support.application.escalation_agent import EscalationAgent
from getnet_support.domain.models import EscalationReason, Locale, Market


def test_br_market_includes_human_contact_channel() -> None:
    agent = EscalationAgent()
    message = agent.handle(
        locale=Locale.PT_BR, reason=EscalationReason.UNKNOWN_CUSTOMER, market=Market.BR
    )
    assert "4002-4000" in message


def test_global_market_omits_br_specific_contact_channel() -> None:
    agent = EscalationAgent()
    message = agent.handle(
        locale=Locale.EN, reason=EscalationReason.UNKNOWN_CUSTOMER, market=Market.GLOBAL
    )
    assert "4002-4000" not in message


def test_different_reasons_produce_different_messages() -> None:
    agent = EscalationAgent()
    unknown_customer = agent.handle(
        locale=Locale.EN, reason=EscalationReason.UNKNOWN_CUSTOMER, market=Market.GLOBAL
    )
    financial_op = agent.handle(
        locale=Locale.EN,
        reason=EscalationReason.UNSUPPORTED_FINANCIAL_OPERATION,
        market=Market.GLOBAL,
    )
    human_request = agent.handle(
        locale=Locale.EN, reason=EscalationReason.EXPLICIT_HUMAN_REQUEST, market=Market.GLOBAL
    )
    assert len({unknown_customer, financial_op, human_request}) == 3
