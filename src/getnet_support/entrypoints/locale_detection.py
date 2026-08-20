"""Heuristic locale detection for requests that do not send an explicit `locale` field."""

import re

from getnet_support.domain.models import Locale

_PT_MARKERS = re.compile(
    r"[ãõáàâêíóôúç]"
    r"|\b(não|você|está|minha|meu|maquininha|conta|vendas|cartão|obrigado|"
    r"como|onde|quando|qual|pelo|pela|para)\b",
    re.IGNORECASE,
)


def detect_locale(message: str) -> Locale:
    """Return `pt-BR` when the message shows Portuguese markers, `en` otherwise."""
    return Locale.PT_BR if _PT_MARKERS.search(message) else Locale.EN
