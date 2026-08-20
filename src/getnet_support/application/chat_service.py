"""Application service that answers one chat turn.

Shared, in-process entry point for both the HTTP API and the UI (REQ-04): the
UI must call this service directly, never over HTTP. The composition root
(the entrypoint) builds and injects every agent; this service never imports
adapters or entrypoint configuration directly (dependency direction).
"""

import time
from uuid import uuid4

from getnet_support.application.agents.customer_support_agent import CustomerSupportAgent
from getnet_support.application.agents.escalation_agent import EscalationAgent
from getnet_support.application.agents.knowledge_agent import KnowledgeAgent
from getnet_support.application.agents.router_agent import RouterAgent
from getnet_support.domain.chat import GroundingOrigin, Language, Market, Route, detect_language

from .dto import ChatResult


class ChatApplicationService:
    """Coordinates agents to answer one chat message."""

    def __init__(
        self,
        *,
        router: RouterAgent,
        knowledge_agent: KnowledgeAgent,
        customer_support_agent: CustomerSupportAgent,
        escalation_agent: EscalationAgent,
    ) -> None:
        """Store the already-composed agents for this process."""
        self._router = router
        self._knowledge_agent = knowledge_agent
        self._customer_support_agent = customer_support_agent
        self._escalation_agent = escalation_agent

    def handle(
        self,
        *,
        message: str,
        user_id: str,
        market: str | None = None,
        locale: str | None = None,
    ) -> ChatResult:
        """Answer one chat turn, degrading honestly when nothing can help yet."""
        started = time.perf_counter()
        language = Language(locale) if locale else detect_language(message)
        decision = self._router.route(message)

        if decision.route is Route.KNOWLEDGE:
            knowledge_result = self._knowledge_agent.answer(
                message, market=Market(market) if market else None, language=language
            )
            answer = knowledge_result.answer
            sources = knowledge_result.sources
            tools: list[str] = []
            grounding = knowledge_result.grounding
            web_search_attempted = knowledge_result.web_search_attempted
            handoff_required = knowledge_result.handoff_required
        elif decision.route is Route.CUSTOMER_SUPPORT:
            support_result = self._customer_support_agent.answer(
                message, user_id=user_id, language=language
            )
            answer = support_result.answer
            sources = []
            tools = support_result.tools
            grounding = support_result.grounding
            web_search_attempted = False
            handoff_required = support_result.handoff_required
        else:
            escalation_result = self._escalation_agent.answer(language=language)
            answer = escalation_result.answer
            sources = []
            tools = []
            grounding = GroundingOrigin.NONE
            web_search_attempted = False
            handoff_required = True

        latency_ms = int((time.perf_counter() - started) * 1000)
        return ChatResult(
            trace_id=str(uuid4()),
            answer=answer,
            language=language,
            route=decision.route,
            agents=["router", decision.route.value],
            tools=tools,
            sources=sources,
            handoff_required=handoff_required,
            grounding=grounding,
            web_search_attempted=web_search_attempted,
            latency_ms=latency_ms,
            decision_source=decision.decision_source,
            classifier_latency_ms=decision.classifier_latency_ms,
        )
