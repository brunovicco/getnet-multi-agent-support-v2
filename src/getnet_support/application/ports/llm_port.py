"""Port for one-shot grounded text generation.

REQ-08: an `LLMPort` implementation is never consulted for authorization,
customer scope, market isolation, or state-changing financial decisions —
only for turning already-approved evidence into prose.
"""

from typing import Protocol


class LLMPort(Protocol):
    """Consumer-defined port for text generation."""

    def generate(self, *, prompt: str, timeout_seconds: float) -> str:
        """Generate text for `prompt` or raise `LLMUnavailableError`."""
        ...
