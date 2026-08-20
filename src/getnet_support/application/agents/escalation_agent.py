"""Escalation Agent: honest human handoff (REQ-08/13).

Reached for state-changing financial requests and prompt-injection attempts
— both decided upstream, deterministically, by the Router Agent (REQ-08).
This agent never fabricates a resolution; it only hands the conversation to
a human.
"""

from dataclasses import dataclass

from getnet_support.domain.chat import Language

_MESSAGE = {
    Language.PT_BR: (
        "Esta solicitação precisa de um atendente humano: ou ela muda o estado da conta, "
        "ou pediu para o assistente ignorar suas instruções — nenhum dos dois casos é algo "
        "que um agente automatizado está autorizado a fazer."
    ),
    Language.EN: (
        "This request needs a human: either it changes account state or it asked the "
        "assistant to ignore its instructions, and neither is something an automated "
        "agent is authorized to do."
    ),
}


@dataclass(frozen=True, slots=True)
class EscalationResult:
    """What the Escalation Agent produced for one query."""

    answer: str


class EscalationAgent:
    """Always hands off to a human; never resolves the request itself."""

    def answer(self, *, language: Language) -> EscalationResult:
        """Return the honest handoff message for `language`."""
        return EscalationResult(answer=_MESSAGE[language])
