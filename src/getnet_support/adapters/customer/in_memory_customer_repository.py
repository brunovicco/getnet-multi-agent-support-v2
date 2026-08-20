"""In-memory customer fixtures, strictly scoped by `user_id` (REQ-17/19).

No I/O, no cross-customer lookup path: each method only ever returns data
for the exact key it was called with.
"""

from decimal import Decimal

from getnet_support.domain.chat import Market
from getnet_support.domain.customer import CustomerProfile, TerminalStatus, Transaction

_PROFILES: dict[str, CustomerProfile] = {
    "cliente1988": CustomerProfile(user_id="cliente1988", name="Loja do Bruno", market=Market.BR),
    "cliente2001": CustomerProfile(
        user_id="cliente2001", name="Mercadinho Central", market=Market.BR
    ),
}

_TRANSACTIONS: dict[str, list[Transaction]] = {
    "cliente1988": [
        Transaction(
            id="tx-9001",
            amount=Decimal("152.40"),
            occurred_at="2026-08-19",
            settles_at="2026-08-21",
            status="settled",
        ),
        Transaction(
            id="tx-9002",
            amount=Decimal("48.00"),
            occurred_at="2026-08-19",
            settles_at="2026-08-21",
            status="settled",
        ),
    ],
    "cliente2001": [
        Transaction(
            id="tx-7001",
            amount=Decimal("980.00"),
            occurred_at="2026-08-19",
            settles_at="2026-08-21",
            status="settled",
        ),
    ],
}

_TERMINALS: dict[str, TerminalStatus] = {
    "cliente1988": TerminalStatus(
        terminal_id="term-1988-A",
        connected=False,
        last_seen_at="2026-08-19T22:10:00Z",
        status_code="S-2074",
    ),
    "cliente2001": TerminalStatus(
        terminal_id="term-2001-A",
        connected=True,
        last_seen_at="2026-08-20T09:00:00Z",
        status_code=None,
    ),
}


class InMemoryCustomerRepository:
    """`CustomerDataPort` implementation backed by fixed, committed fixtures."""

    def get_customer_profile(self, user_id: str) -> CustomerProfile | None:
        """Return the profile for `user_id`, or None if unknown."""
        return _PROFILES.get(user_id)

    def get_recent_transactions(self, user_id: str) -> list[Transaction] | None:
        """Return recent transactions for `user_id`, or None if unknown."""
        if user_id not in _PROFILES:
            return None
        return list(_TRANSACTIONS.get(user_id, []))

    def get_terminal_status(self, user_id: str) -> TerminalStatus | None:
        """Return terminal status for `user_id`, or None if unknown."""
        if user_id not in _PROFILES:
            return None
        return _TERMINALS.get(user_id)
