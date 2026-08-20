"""Unit tests for message language detection (REQ-20)."""

import pytest

from getnet_support.domain.chat import Language, detect_language


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Qual a diferença entre a Get Clássica e a Get Smart?", Language.PT_BR),
        ("What's the difference between the Get Clássica and the Get Smart?", Language.EN),
        ("Minha maquininha não está pegando sinal, o que eu faço?", Language.PT_BR),
        ("My card machine won't connect to the internet, what should I do?", Language.EN),
        ("Como está o tempo hoje?", Language.PT_BR),
    ],
)
def test_detect_language(message: str, expected: Language) -> None:
    """Detection must be correct for both official-scenario phrasings."""
    assert detect_language(message) is expected
