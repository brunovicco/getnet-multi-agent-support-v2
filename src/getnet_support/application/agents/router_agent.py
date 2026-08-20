"""Router Agent: deterministic, rule-first routing (REQ-05/06/07/08).

Routing to `customer_support` or `escalation` is decided by intent-phrase
rules that identify *who should handle this*, never by asking "does this
message need the web" — that decision belongs solely to the Knowledge
Agent's evidence gate (REQ-06). Nothing here is authoritative for
authorization, customer scope, market isolation, or a state-changing
financial operation (REQ-08): those are always rule-only, evaluated before
any optional LLM tie-break could run.
"""

import re
import time
from dataclasses import dataclass

from getnet_support.domain.chat import DecisionSource, Route

_ESCALATION_PATTERNS = (
    re.compile(r"\btransfir\w*\b", re.IGNORECASE),
    re.compile(r"\btransfer\w*\b", re.IGNORECASE),
    re.compile(r"ignore\s+(your|these|all)\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(as\s+|suas\s+)?instru[cç][oõ]es", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
)

_CUSTOMER_SUPPORT_PATTERNS = (
    re.compile(r"\bmaquininha\b", re.IGNORECASE),
    re.compile(r"\bcard\s+machine\b", re.IGNORECASE),
    re.compile(r"\bterminal\b", re.IGNORECASE),
    re.compile(r"\bcai\s+o\s+dinheiro\b", re.IGNORECASE),
    re.compile(r"\bvenda\s+de\s+ontem\b", re.IGNORECASE),
    re.compile(r"yesterday'?s\s+sales", re.IGNORECASE),
    re.compile(r"money\b.*\bdeposited\b", re.IGNORECASE),
    re.compile(r"transaction\s+decline", re.IGNORECASE),
    re.compile(r"decline\s+error", re.IGNORECASE),
    re.compile(r"erro\s+de\s+transa[cç][aã]o", re.IGNORECASE),
    re.compile(r"n[ãa]o\s+(est[áa]\s+)?pegando\s+sinal", re.IGNORECASE),
    re.compile(r"won'?t\s+connect", re.IGNORECASE),
    re.compile(r"\btransa[cç][oõ]es\b", re.IGNORECASE),
    re.compile(r"\btransactions?\b", re.IGNORECASE),
)

_RULE_CONFIDENCE = {
    Route.ESCALATION: 1.0,
    Route.CUSTOMER_SUPPORT: 0.9,
}
_DEFAULT_CONFIDENCE = 0.55


@dataclass(frozen=True, slots=True)
class RouterDecision:
    """The outcome of one routing decision."""

    route: Route
    decision_source: DecisionSource
    classifier_latency_ms: int


class RouterAgent:
    """Decides which agent owns a chat turn, rules first (REQ-07)."""

    def __init__(self, *, confidence_min: float = 0.6) -> None:
        """Store the confidence floor below which a rule is not decisive."""
        self._confidence_min = confidence_min

    def route(self, message: str) -> RouterDecision:
        """Classify one message deterministically, timing the decision."""
        started = time.perf_counter()
        route, confidence = self._match(message)
        source = (
            DecisionSource.RULE if confidence >= self._confidence_min else DecisionSource.FALLBACK
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return RouterDecision(route=route, decision_source=source, classifier_latency_ms=latency_ms)

    def _match(self, message: str) -> tuple[Route, float]:
        """Return the best-matching route and how confident the rule set is."""
        if any(pattern.search(message) for pattern in _ESCALATION_PATTERNS):
            return Route.ESCALATION, _RULE_CONFIDENCE[Route.ESCALATION]
        if any(pattern.search(message) for pattern in _CUSTOMER_SUPPORT_PATTERNS):
            return Route.CUSTOMER_SUPPORT, _RULE_CONFIDENCE[Route.CUSTOMER_SUPPORT]
        return Route.KNOWLEDGE, _DEFAULT_CONFIDENCE
