"""Port for customer-scoped data lookups (REQ-17/18/19).

Every method takes the *authenticated* `user_id` only — never an identifier
parsed out of message text. Isolation between customers is enforced by
never calling this port with anything other than that authenticated id.
"""

from typing import Protocol

from getnet_support.domain.customer import CustomerProfile, TerminalStatus, Transaction


class CustomerDataPort(Protocol):
    """Consumer-defined port for the three required customer tools."""

    def get_customer_profile(self, user_id: str) -> CustomerProfile | None:
        """Return the profile for `user_id`, or None if unknown."""
        ...

    def get_recent_transactions(self, user_id: str) -> list[Transaction] | None:
        """Return recent transactions for `user_id`, or None if unknown."""
        ...

    def get_terminal_status(self, user_id: str) -> TerminalStatus | None:
        """Return terminal status for `user_id`, or None if unknown."""
        ...
