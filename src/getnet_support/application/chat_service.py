"""Chat Application Service: orchestrates the Router and specialized agents.

This is the single entry point shared by the HTTP API and the Gradio UI — both call
`ChatApplicationService.handle` on the same instance, so there is no internal HTTP hop between the
UI and the backend.
"""

import time
import uuid

from getnet_support.application.customer_support_agent import (
    CustomerSupportAgent,
    CustomerSupportResult,
)
from getnet_support.application.escalation_agent import EscalationAgent
from getnet_support.application.guardrails import is_unsupported_financial_operation
from getnet_support.application.knowledge_agent import KnowledgeAgent
from getnet_support.application.ports import CustomerNotFoundError
from getnet_support.application.router_agent import RouterAgent
from getnet_support.domain.models import (
    AgentName,
    ChatResult,
    EscalationReason,
    Locale,
    Market,
    Route,
    Source,
    TerminalStatus,
)

_TERMINAL_TROUBLESHOOTING_QUERY = (
    "maquininha terminal sem internet nao conecta offline troubleshooting"
)


class ChatApplicationService:
    """Single entry point shared by the HTTP API and the Gradio UI."""

    def __init__(
        self,
        router: RouterAgent,
        knowledge_agent: KnowledgeAgent,
        customer_support_agent: CustomerSupportAgent,
        escalation_agent: EscalationAgent,
    ) -> None:
        """Bind the Router and the three specialized agents it coordinates."""
        self._router = router
        self._knowledge_agent = knowledge_agent
        self._customer_support_agent = customer_support_agent
        self._escalation_agent = escalation_agent

    async def handle(
        self, *, message: str, user_id: str, market: Market, locale: Locale
    ) -> ChatResult:
        """Route and answer one chat message; never raises for expected failure modes."""
        started = time.monotonic()
        trace_id = uuid.uuid4().hex
        route = self._router.route(message)
        agents = [AgentName.ROUTER]
        tools: list[str] = []
        sources: tuple[Source, ...] = ()
        handoff_required = False

        if route is Route.ESCALATION:
            agents.append(AgentName.ESCALATION)
            escalation_reason = (
                EscalationReason.UNSUPPORTED_FINANCIAL_OPERATION
                if is_unsupported_financial_operation(message)
                else EscalationReason.EXPLICIT_HUMAN_REQUEST
            )
            answer = self._escalation_agent.handle(
                locale=locale, reason=escalation_reason, market=market
            )
            handoff_required = True
        elif route is Route.CUSTOMER_SUPPORT:
            agents.append(AgentName.CUSTOMER_SUPPORT)
            answer, chained_tools, chained_sources, handoff_required = await self._handle_support(
                message=message, user_id=user_id, market=market, locale=locale, agents=agents
            )
            tools.extend(chained_tools)
            sources = chained_sources
        else:
            agents.append(AgentName.KNOWLEDGE)
            knowledge_result = await self._knowledge_agent.handle(
                message=message, market=market, locale=locale
            )
            tools.append(knowledge_result.tool_used)
            sources = knowledge_result.sources
            answer = knowledge_result.answer
            if not knowledge_result.sufficient_evidence:
                agents.append(AgentName.ESCALATION)
                handoff_required = True

        latency_ms = int((time.monotonic() - started) * 1000)
        return ChatResult(
            answer=answer,
            sources=sources,
            route=route,
            agents=tuple(agents),
            tools=tuple(tools),
            handoff_required=handoff_required,
            trace_id=trace_id,
            latency_ms=latency_ms,
        )

    async def _handle_support(
        self,
        *,
        message: str,
        user_id: str,
        market: Market,
        locale: Locale,
        agents: list[AgentName],
    ) -> tuple[str, list[str], tuple[Source, ...], bool]:
        try:
            result = self._customer_support_agent.handle(user_id=user_id, message=message)
        except CustomerNotFoundError:
            agents.append(AgentName.ESCALATION)
            answer = self._escalation_agent.handle(
                locale=locale, reason=EscalationReason.UNKNOWN_CUSTOMER, market=market
            )
            return answer, [], (), True

        tools = list(result.tools_used)
        answer = _render_customer_support_answer(result, locale)
        sources: tuple[Source, ...] = ()

        if result.terminal_status is not None:
            agents.append(AgentName.KNOWLEDGE)
            knowledge_result = await self._knowledge_agent.handle(
                message=_TERMINAL_TROUBLESHOOTING_QUERY, market=market, locale=locale
            )
            tools.append(knowledge_result.tool_used)
            sources = knowledge_result.sources
            answer = f"{answer}\n\n{knowledge_result.answer}"

        return answer, tools, sources, False


def _render_customer_support_answer(result: CustomerSupportResult, locale: Locale) -> str:
    lines: list[str] = []
    name, plan = result.profile.name, result.profile.plan
    if locale is Locale.PT_BR:
        lines.append(f"Oi {name}, aqui está o que encontrei na sua conta ({plan}):")
    else:
        lines.append(f"Hi {name}, here is what I found on your account ({plan}):")

    for tx in result.transactions or ():
        if locale is Locale.PT_BR:
            lines.append(
                f"- Venda de {tx.occurred_at} no valor de {tx.currency} {tx.amount}: "
                f"status {tx.status}, liquidação prevista para {tx.settlement_date}."
            )
        else:
            lines.append(
                f"- Sale on {tx.occurred_at} for {tx.currency} {tx.amount}: "
                f"status {tx.status}, settlement expected on {tx.settlement_date}."
            )

    if result.terminal_status is not None:
        lines.append(_render_terminal_status(result.terminal_status, locale))

    return "\n".join(lines)


def _render_terminal_status(status: TerminalStatus, locale: Locale) -> str:
    state_pt = "online" if status.online else "offline"
    if locale is Locale.PT_BR:
        error = f", código de erro {status.error_code}." if status.error_code else "."
        return (
            f"- Terminal {status.terminal_id}: {state_pt}, "
            f"última conexão {status.last_seen_at}{error}"
        )
    error = f", error code {status.error_code}." if status.error_code else "."
    return f"- Terminal {status.terminal_id}: {state_pt}, last seen {status.last_seen_at}{error}"
