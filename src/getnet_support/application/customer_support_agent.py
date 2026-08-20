"""Customer Support Agent: retrieves customer-scoped data via deterministic tools.

The LLM is never involved in this agent: `user_id` always comes from the authenticated request,
never from message text, and tool output is rendered with fixed templates so no customer fact can
be fabricated.
"""

import re
from dataclasses import dataclass

from getnet_support.application.ports import CustomerDataPort
from getnet_support.domain.models import CustomerProfile, TerminalStatus, Transaction

_TERMINAL_PATTERN = re.compile(
    r"\b(maquininha|terminal|conex[ãa]o|n[ãa]o\s+conecta|offline|internet)\b", re.IGNORECASE
)
_TRANSACTION_PATTERN = re.compile(
    r"\b(dep[óo]sito|deposited|vendas|sales|recebimento|settlement|extrato|dinheiro)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CustomerSupportResult:
    """Structured result of one customer support lookup."""

    profile: CustomerProfile
    transactions: tuple[Transaction, ...] | None
    terminal_status: TerminalStatus | None
    tools_used: tuple[str, ...]


class CustomerSupportAgent:
    """Answers customer-specific questions using deterministic tools only."""

    def __init__(self, customer_data: CustomerDataPort) -> None:
        """Bind the customer-scoped data port used by this agent's tools."""
        self._customer_data = customer_data

    def handle(self, *, user_id: str, message: str) -> CustomerSupportResult:
        """Look up the customer-scoped data relevant to the message.

        Propagates CustomerNotFoundError from the port when `user_id` is unknown; the caller is
        responsible for escalating in that case.
        """
        profile = self._customer_data.get_customer_profile(user_id)
        tools_used = ["get_customer_profile"]

        wants_terminal = bool(_TERMINAL_PATTERN.search(message))
        wants_transactions = bool(_TRANSACTION_PATTERN.search(message)) or not wants_terminal

        terminal_status: TerminalStatus | None = None
        if wants_terminal:
            terminal_status = self._customer_data.get_terminal_status(user_id)
            tools_used.append("get_terminal_status")

        transactions: tuple[Transaction, ...] | None = None
        if wants_transactions:
            transactions = self._customer_data.get_recent_transactions(user_id)
            tools_used.append("get_recent_transactions")

        return CustomerSupportResult(
            profile=profile,
            transactions=transactions,
            terminal_status=terminal_status,
            tools_used=tuple(tools_used),
        )
