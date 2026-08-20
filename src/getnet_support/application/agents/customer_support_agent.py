"""Customer Support Agent: deterministic, customer-scoped tool use.

REQ-17/18/19: every tool call uses the *authenticated* `user_id` only. A
customer id parsed out of the message text is never used to look up data —
if the message names a different customer, this agent escalates instead of
answering, before any tool is called.
"""

import re
from dataclasses import dataclass

from getnet_support.application.ports.customer_data_port import CustomerDataPort
from getnet_support.domain.chat import GroundingOrigin, Language
from getnet_support.domain.customer import CustomerProfile, TerminalStatus, Transaction

_FOREIGN_CUSTOMER_RE = re.compile(r"\bcliente\d+\b", re.IGNORECASE)

_TERMINAL_PATTERNS = (
    re.compile(r"\bmaquininha\b", re.IGNORECASE),
    re.compile(r"\bcard\s+machine\b", re.IGNORECASE),
    re.compile(r"\bterminal\b", re.IGNORECASE),
    re.compile(r"\bsinal\b", re.IGNORECASE),
    re.compile(r"\bconnect", re.IGNORECASE),
    re.compile(r"\bdecline", re.IGNORECASE),
    re.compile(r"\berro\b", re.IGNORECASE),
)
_TRANSACTION_PATTERNS = (
    re.compile(r"\bvenda\b", re.IGNORECASE),
    re.compile(r"\bsale[s]?\b", re.IGNORECASE),
    re.compile(r"\bdinheiro\b", re.IGNORECASE),
    re.compile(r"\bmoney\b", re.IGNORECASE),
    re.compile(r"\bdeposit", re.IGNORECASE),
    re.compile(r"\btransa[cç][oõ]es\b", re.IGNORECASE),
    re.compile(r"\btransactions?\b", re.IGNORECASE),
)

_UNKNOWN_USER_MESSAGE = {
    Language.PT_BR: "Não localizei sua conta. Um atendente humano vai confirmar seus dados.",
    Language.EN: "I could not find your account. A human agent will confirm your details.",
}
_CROSS_CUSTOMER_MESSAGE = {
    Language.PT_BR: "Só posso mostrar dados da conta autenticada nesta conversa.",
    Language.EN: "I can only show data for the account authenticated in this conversation.",
}


@dataclass(frozen=True, slots=True)
class CustomerSupportResult:
    """What the Customer Support Agent produced for one query.

    `chain_to_knowledge` (P1.3) signals that a real problem was found (e.g.
    a disconnected terminal) that a matching KB troubleshooting article
    could supplement. The orchestrator (`ChatApplicationService`) decides
    whether to actually chain — this agent has no Knowledge Agent
    dependency, only the signal.
    """

    answer: str
    tools: list[str]
    grounding: GroundingOrigin
    handoff_required: bool
    chain_to_knowledge: bool = False


class CustomerSupportAgent:
    """Resolves account-specific questions using customer-scoped tools."""

    def __init__(self, customer_data: CustomerDataPort) -> None:
        """Wire the customer data port."""
        self._customer_data = customer_data

    def answer(self, message: str, *, user_id: str, language: Language) -> CustomerSupportResult:
        """Answer one support question, scoped strictly to `user_id`."""
        foreign_match = _FOREIGN_CUSTOMER_RE.search(message)
        if foreign_match and foreign_match.group(0).lower() != user_id.lower():
            return CustomerSupportResult(
                answer=_CROSS_CUSTOMER_MESSAGE[language],
                tools=[],
                grounding=GroundingOrigin.NONE,
                handoff_required=True,
            )

        profile = self._customer_data.get_customer_profile(user_id)
        if profile is None:
            return CustomerSupportResult(
                answer=_UNKNOWN_USER_MESSAGE[language],
                tools=[],
                grounding=GroundingOrigin.NONE,
                handoff_required=True,
            )

        if any(pattern.search(message) for pattern in _TERMINAL_PATTERNS):
            status = self._customer_data.get_terminal_status(user_id)
            has_a_real_problem = status is not None and not status.connected
            return CustomerSupportResult(
                answer=self._describe_terminal(status, language=language),
                tools=["get_terminal_status"],
                grounding=GroundingOrigin.CUSTOMER_DATA,
                handoff_required=False,
                chain_to_knowledge=has_a_real_problem,
            )

        if any(pattern.search(message) for pattern in _TRANSACTION_PATTERNS):
            transactions = self._customer_data.get_recent_transactions(user_id)
            return CustomerSupportResult(
                answer=self._describe_transactions(transactions, language=language),
                tools=["get_recent_transactions"],
                grounding=GroundingOrigin.CUSTOMER_DATA,
                handoff_required=False,
            )

        return CustomerSupportResult(
            answer=self._describe_profile(profile, language=language),
            tools=["get_customer_profile"],
            grounding=GroundingOrigin.CUSTOMER_DATA,
            handoff_required=False,
        )

    def _describe_terminal(self, status: TerminalStatus | None, *, language: Language) -> str:
        """Render `get_terminal_status` output deterministically."""
        if status is None or status.connected:
            return {
                Language.PT_BR: "Sua maquininha está conectada e sem alertas no momento.",
                Language.EN: "Your card machine is connected with no active alerts.",
            }[language]
        code = f" ({status.status_code})" if status.status_code else ""
        return {
            Language.PT_BR: (
                f"Sua maquininha {status.terminal_id} está sem conexão desde "
                f"{status.last_seen_at}{code}. Reconecte ao Wi-Fi ou verifique o chip de "
                "dados; se persistir, acione o suporte técnico."
            ),
            Language.EN: (
                f"Your card machine {status.terminal_id} has been disconnected since "
                f"{status.last_seen_at}{code}. Reconnect it to Wi-Fi or check its data "
                "chip; contact technical support if it persists."
            ),
        }[language]

    def _describe_transactions(
        self, transactions: list[Transaction] | None, *, language: Language
    ) -> str:
        """Render `get_recent_transactions` output deterministically."""
        if not transactions:
            return {
                Language.PT_BR: "Não encontrei vendas recentes na sua conta.",
                Language.EN: "I could not find recent sales on your account.",
            }[language]
        latest = transactions[0]
        return {
            Language.PT_BR: (
                f"Sua venda mais recente ({latest.id}, R$ {latest.amount}) foi em "
                f"{latest.occurred_at} e o valor cai em {latest.settles_at}."
            ),
            Language.EN: (
                f"Your most recent sale ({latest.id}, R$ {latest.amount}) happened on "
                f"{latest.occurred_at} and settles on {latest.settles_at}."
            ),
        }[language]

    def _describe_profile(self, profile: CustomerProfile, *, language: Language) -> str:
        """Render `get_customer_profile` output deterministically."""
        return {
            Language.PT_BR: f"Encontrei sua conta ({profile.name}). Como posso ajudar?",
            Language.EN: f"I found your account ({profile.name}). How can I help?",
        }[language]
