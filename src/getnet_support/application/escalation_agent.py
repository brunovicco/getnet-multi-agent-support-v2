"""Escalation Agent: returns a safe, non-fabricated handoff message."""

from getnet_support.domain.models import Locale

_MESSAGES = {
    Locale.PT_BR: (
        "Não consigo resolver essa solicitação com segurança agora. "
        "Um atendente humano vai continuar o seu atendimento."
    ),
    Locale.EN: (
        "I can't safely resolve this request right now. A human agent will take over from here."
    ),
}


class EscalationAgent:
    """Produces the handoff response for unknown users, low confidence, or unsupported requests."""

    def handle(self, *, locale: Locale, reason: str) -> str:
        """Return the handoff message for the given locale; reason is for logging, not display."""
        return _MESSAGES[locale]
