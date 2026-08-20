"""Escalation Agent: returns a safe, non-fabricated, reason-specific handoff message.

Never fabricates a resolution — every branch here is a "stop and hand off" outcome. The BR contact
channel quoted below (4002-4000) is the real phone number cited on Getnet's official troubleshooting
page (see `adapters/corpus/getnet_br.json`, chunk `br-troubleshooting-erros`), not invented.
"""

from getnet_support.domain.models import EscalationReason, Locale, Market

_BR_CONTACT_HINT = {
    Locale.PT_BR: " Você também pode ligar para 4002-4000 ou chamar no WhatsApp da Getnet.",
    Locale.EN: " You can also call 4002-4000 or reach Getnet on WhatsApp.",
}

_MESSAGES: dict[EscalationReason, dict[Locale, str]] = {
    EscalationReason.UNKNOWN_CUSTOMER: {
        Locale.PT_BR: (
            "Não encontrei uma conta com esse identificador, então não posso acessar dados de "
            "cliente com segurança. Um atendente humano vai continuar o seu atendimento."
        ),
        Locale.EN: (
            "I couldn't find an account for that identifier, so I can't safely access customer "
            "data. A human agent will take over from here."
        ),
    },
    EscalationReason.UNSUPPORTED_FINANCIAL_OPERATION: {
        Locale.PT_BR: (
            "Estorno, cancelamento de conta e contestação de compra exigem verificação adicional "
            "que não posso concluir por aqui. Um atendente humano vai continuar o seu atendimento."
        ),
        Locale.EN: (
            "Refunds, account cancellations, and purchase disputes need additional verification "
            "I can't complete here. A human agent will take over from here."
        ),
    },
    EscalationReason.EXPLICIT_HUMAN_REQUEST: {
        Locale.PT_BR: "Sem problema — vou te transferir para um atendente humano agora.",
        Locale.EN: "No problem — I'm transferring you to a human agent now.",
    },
}


class EscalationAgent:
    """Produces a reason-specific handoff response; never resolves the request itself."""

    def handle(self, *, locale: Locale, reason: EscalationReason, market: Market) -> str:
        """Return the handoff message for the given locale and reason."""
        message = _MESSAGES[reason][locale]
        if market is Market.BR:
            message += _BR_CONTACT_HINT[locale]
        return message
