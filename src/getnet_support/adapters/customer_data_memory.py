"""In-memory customer data adapter simulating CRM, settlement, and terminal management systems.

Fixture data only — no production personal data. Seeded for the two demo users required by the
challenge: `cliente1988` (terminal currently offline) and `cliente2001` (terminal online).
"""

from dataclasses import dataclass
from decimal import Decimal

from getnet_support.application.ports import CustomerDataPort, CustomerNotFoundError
from getnet_support.domain.models import CustomerProfile, Market, TerminalStatus, Transaction


@dataclass(frozen=True, slots=True)
class _CustomerRecord:
    """Fixture bundle for one fake customer."""

    profile: CustomerProfile
    transactions: tuple[Transaction, ...]
    terminal_status: TerminalStatus


_CUSTOMERS: dict[str, _CustomerRecord] = {
    "cliente1988": _CustomerRecord(
        profile=CustomerProfile(
            user_id="cliente1988", name="Ana", plan="Get Smart", market=Market.BR
        ),
        transactions=(
            Transaction(
                transaction_id="tx-1001",
                occurred_at="2026-08-18",
                amount=Decimal("482.50"),
                currency="BRL",
                settlement_date="2026-08-19",
                status="settled",
            ),
        ),
        terminal_status=TerminalStatus(
            terminal_id="term-8842",
            online=False,
            last_seen_at="2026-08-18T22:14:00Z",
            error_code="NET_TIMEOUT",
        ),
    ),
    "cliente2001": _CustomerRecord(
        profile=CustomerProfile(
            user_id="cliente2001", name="Bruno", plan="Get Clássica", market=Market.BR
        ),
        transactions=(
            Transaction(
                transaction_id="tx-2001",
                occurred_at="2026-08-18",
                amount=Decimal("129.90"),
                currency="BRL",
                settlement_date="2026-08-19",
                status="settled",
            ),
        ),
        terminal_status=TerminalStatus(
            terminal_id="term-2091",
            online=True,
            last_seen_at="2026-08-19T09:02:00Z",
            error_code=None,
        ),
    ),
}


class InMemoryCustomerDataAdapter(CustomerDataPort):
    """Fake CRM + settlement + terminal management backend for demo and tests."""

    def get_customer_profile(self, user_id: str) -> CustomerProfile:
        """Return the customer profile or raise CustomerNotFoundError."""
        return self._record(user_id).profile

    def get_recent_transactions(self, user_id: str) -> tuple[Transaction, ...]:
        """Return recent settlement transactions or raise CustomerNotFoundError."""
        return self._record(user_id).transactions

    def get_terminal_status(self, user_id: str) -> TerminalStatus:
        """Return terminal connectivity status or raise CustomerNotFoundError."""
        return self._record(user_id).terminal_status

    @staticmethod
    def _record(user_id: str) -> _CustomerRecord:
        record = _CUSTOMERS.get(user_id)
        if record is None:
            raise CustomerNotFoundError(user_id)
        return record
