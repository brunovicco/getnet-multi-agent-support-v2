"""Core chat Value Objects shared by every agent and boundary.

No framework, transport, or SDK types belong here — see
``.claude/rules/architecture.md``.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Short function words that are common in both languages and roughly equally
# frequent either way; kept in both sets so they cancel out instead of
# skewing the vote, leaving the unambiguous words to decide (REQ-20).
_AMBIGUOUS_WORDS = frozenset({"a", "e", "o", "i", "in", "de"})
_PT_WORDS = (
    frozenset(
        {
            "qual",
            "quando",
            "como",
            "quem",
            "onde",
            "porque",
            "por",
            "para",
            "pra",
            "com",
            "sem",
            "sobre",
            "você",
            "voce",
            "não",
            "nao",
            "está",
            "esta",
            "são",
            "sao",
            "tem",
            "minha",
            "meu",
            "vai",
            "dinheiro",
            "venda",
            "vendas",
            "maquininha",
            "hoje",
            "ontem",
            "amanhã",
            "amanha",
            "cai",
            "pegando",
            "sinal",
            "cotação",
            "cotacao",
            "dá",
            "da",
            "do",
            "das",
            "dos",
            "às",
            "então",
            "preciso",
            "posso",
        }
    )
    | _AMBIGUOUS_WORDS
)
_EN_WORDS = (
    frozenset(
        {
            "the",
            "an",
            "of",
            "on",
            "at",
            "to",
            "for",
            "and",
            "or",
            "is",
            "are",
            "does",
            "do",
            "my",
            "your",
            "you",
            "with",
            "what",
            "when",
            "how",
            "can",
            "will",
            "using",
            "through",
            "into",
            "money",
            "sales",
            "sale",
            "machine",
            "card",
            "connect",
            "decline",
            "error",
            "need",
            "today",
            "tomorrow",
            "yesterday",
        }
    )
    | _AMBIGUOUS_WORDS
)


class Route(StrEnum):
    """Which agent owns a chat turn."""

    KNOWLEDGE = "knowledge"
    CUSTOMER_SUPPORT = "customer_support"
    ESCALATION = "escalation"


class Language(StrEnum):
    """Response language, independent of market (REQ-21)."""

    PT_BR = "pt-BR"
    EN = "en"


class Market(StrEnum):
    """Commercial market whose corpus content applies."""

    BR = "BR"
    GLOBAL = "GLOBAL"


class SourceOrigin(StrEnum):
    """Where a cited source came from."""

    GETNET_KB = "getnet_kb"
    WEB = "web"


class Volatility(StrEnum):
    """How likely a source's content is to change.

    The committed corpus (curated in an earlier round, not touched here) uses
    a three-level scale; REQ-02 only enumerates ``low``/``high`` as examples,
    so ``medium`` is preserved as-is rather than collapsed.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GroundingOrigin(StrEnum):
    """The evidence source actually used to ground an answer."""

    GETNET_KB = "getnet_kb"
    WEB = "web"
    CUSTOMER_DATA = "customer_data"
    NONE = "none"


class DecisionSource(StrEnum):
    """How a routing decision was made (REQ-07)."""

    RULE = "rule"
    LLM = "llm"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class Source:
    """One citation attached to a chat answer."""

    title: str
    url: str
    origin: SourceOrigin
    retrieved_at: str
    volatility: Volatility
    market: Market | None = None


def detect_language(text: str) -> Language:
    """Detect PT-BR vs EN from message text (REQ-20 fallback when no `locale`)."""
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    pt_score = sum(1 for token in tokens if token in _PT_WORDS)
    en_score = sum(1 for token in tokens if token in _EN_WORDS)
    return Language.PT_BR if pt_score > en_score else Language.EN
