"""Runs the minimal evaluation dataset as a regression test against the wired system.

`evaluation/scenarios.json` mirrors the challenge brief's own example test scenarios; this test
guards the Router/agent wiring against silently misrouting one of them (see the English
"card machine won't connect" regression this caught during implementation).
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from getnet_support.adapters.customer_data_memory import InMemoryCustomerDataAdapter
from getnet_support.adapters.knowledge_retriever_local import LocalKnowledgeRetriever
from getnet_support.application.chat_service import ChatApplicationService
from getnet_support.application.customer_support_agent import CustomerSupportAgent
from getnet_support.application.escalation_agent import EscalationAgent
from getnet_support.application.knowledge_agent import KnowledgeAgent
from getnet_support.application.ports import LLMGenerationError
from getnet_support.application.router_agent import RouterAgent
from getnet_support.domain.models import Locale, Market, WebSearchResult

_SCENARIOS_PATH = Path(__file__).resolve().parents[2] / "evaluation" / "scenarios.json"


class _RaisingLLM:
    """Always fails, forcing the deterministic extractive fallback for every scenario."""

    async def generate(self, *, system_prompt: str, user_prompt: str, locale: Locale) -> str:
        """Simulate no LLM provider being configured."""
        raise LLMGenerationError("no provider configured in eval run")


class _AlwaysConfiguredWebSearch:
    """Returns one canned result for every query, like a healthy Tavily integration."""

    def is_configured(self) -> bool:
        """Report as configured."""
        return True

    async def search(self, query: str) -> tuple[WebSearchResult, ...]:
        """Return a single canned result regardless of the query."""
        return (WebSearchResult(title="Result", url="https://example.test", snippet="Info"),)


def _load_scenarios() -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(_SCENARIOS_PATH.read_text(encoding="utf-8"))
    return data


def _build_service() -> ChatApplicationService:
    return ChatApplicationService(
        router=RouterAgent(),
        knowledge_agent=KnowledgeAgent(
            retriever=LocalKnowledgeRetriever(),
            web_search=_AlwaysConfiguredWebSearch(),
            llm=_RaisingLLM(),
        ),
        customer_support_agent=CustomerSupportAgent(InMemoryCustomerDataAdapter()),
        escalation_agent=EscalationAgent(),
    )


@pytest.mark.parametrize("scenario", _load_scenarios(), ids=lambda s: s["id"])
def test_official_scenario_routes_and_resolves_as_expected(scenario: dict[str, Any]) -> None:
    service = _build_service()
    result = asyncio.run(
        service.handle(
            message=scenario["message"],
            user_id=scenario["user_id"],
            market=Market.BR,
            locale=Locale.EN,
        )
    )
    assert result.route.value == scenario["expected_route"]
    if "expected_tool" in scenario:
        assert scenario["expected_tool"] in result.tools
    if scenario.get("expect_sources"):
        assert result.sources
    assert result.handoff_required == scenario["expect_handoff"]
