"""Customer-scoped domain Value Objects for the Customer Support Agent."""

from dataclasses import dataclass
from decimal import Decimal

from getnet_support.domain.chat import Market


@dataclass(frozen=True, slots=True)
class CustomerProfile:
    """The identity of one authenticated customer."""

    user_id: str
    name: str
    market: Market


@dataclass(frozen=True, slots=True)
class Transaction:
    """One settled or pending sale for a customer."""

    id: str
    amount: Decimal
    occurred_at: str
    settles_at: str
    status: str


@dataclass(frozen=True, slots=True)
class TerminalStatus:
    """The connectivity state of a customer's card machine."""

    terminal_id: str
    connected: bool
    last_seen_at: str
    status_code: str | None
