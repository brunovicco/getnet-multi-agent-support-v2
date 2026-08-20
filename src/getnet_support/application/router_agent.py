"""Router Agent: deterministic classification of inbound requests.

Never delegates authorization, guardrails, customer data access, market isolation, or financial
operations to an LLM. Ambiguous product-vs-general questions default to the Knowledge Agent, which
owns the further RAG-vs-web-search decision.
"""

import re

from getnet_support.application.guardrails import is_unsupported_financial_operation
from getnet_support.domain.models import Route

_CUSTOMER_SUPPORT_PATTERN = re.compile(
    r"\b("
    r"maquininha|terminal|card\s+machine|conex[ãa]o|n[ãa]o\s+conecta|"
    r"(won.?t|can.?t|not)\s+connect|connect(ing)?\s+to\s+the\s+internet|offline|"
    r"transa[çc][ãa]o|recus\w*|declin\w*|"
    r"dep[óo]sito|deposited|vendas\s+de\s+ontem|yesterday.?s\s+sales|"
    r"minha\s+conta|my\s+account|extrato|settlement|recebimento"
    r")\b",
    re.IGNORECASE,
)

_ESCALATION_PATTERN = re.compile(
    r"\b(falar\s+com\s+(um\s+)?(atendente|humano)|human\s+agent|"
    r"talk\s+to\s+(a\s+)?(human|agent)|quero\s+um\s+atendente)\b",
    re.IGNORECASE,
)


class RouterAgent:
    """Classifies one inbound message into a Route using deterministic rules."""

    def route(self, message: str) -> Route:
        """Return the destination for a message; escalation signals win over other matches."""
        if _ESCALATION_PATTERN.search(message) or is_unsupported_financial_operation(message):
            return Route.ESCALATION
        if _CUSTOMER_SUPPORT_PATTERN.search(message):
            return Route.CUSTOMER_SUPPORT
        return Route.KNOWLEDGE
