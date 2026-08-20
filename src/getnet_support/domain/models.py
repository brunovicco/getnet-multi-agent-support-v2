"""Domain value objects for the Getnet multi-agent support system.

Pure business concepts: immutable, framework-free, and shared by application ports and agents.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class Market(StrEnum):
    """Commercial market whose facts may be used to answer a request."""

    BR = "BR"
    GLOBAL = "GLOBAL"


class Locale(StrEnum):
    """Response language."""

    PT_BR = "pt-BR"
    EN = "en"


class AgentName(StrEnum):
    """Agent identifiers surfaced in execution traces."""

    ROUTER = "router"
    KNOWLEDGE = "knowledge"
    CUSTOMER_SUPPORT = "customer_support"
    ESCALATION = "escalation"


class Route(StrEnum):
    """Destination chosen by the Router Agent."""

    KNOWLEDGE = "knowledge"
    CUSTOMER_SUPPORT = "customer_support"
    ESCALATION = "escalation"


@dataclass(frozen=True, slots=True)
class Source:
    """One cited knowledge source shown to the user."""

    title: str
    url: str
    market: Market
    retrieved_at: str
    volatility: str


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    """One persisted knowledge chunk with required provenance metadata."""

    id: str
    text: str
    title: str
    source: str
    market: Market
    language: Locale
    topic: str
    retrieved_at: str
    volatility: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A knowledge chunk paired with its retrieval score."""

    chunk: KnowledgeChunk
    score: float


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    """One web search result returned by an external search provider."""

    title: str
    url: str
    snippet: str


@dataclass(frozen=True, slots=True)
class CustomerProfile:
    """Customer-scoped profile data."""

    user_id: str
    name: str
    plan: str
    market: Market


@dataclass(frozen=True, slots=True)
class Transaction:
    """One customer settlement transaction."""

    transaction_id: str
    occurred_at: str
    amount: Decimal
    currency: str
    settlement_date: str
    status: str


@dataclass(frozen=True, slots=True)
class TerminalStatus:
    """Connectivity/status snapshot for a customer's card terminal."""

    terminal_id: str
    online: bool
    last_seen_at: str
    error_code: str | None


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Outcome of handling one chat request, returned by the application service."""

    answer: str
    sources: tuple[Source, ...]
    route: Route
    agents: tuple[AgentName, ...]
    tools: tuple[str, ...]
    handoff_required: bool
    trace_id: str
    latency_ms: int
