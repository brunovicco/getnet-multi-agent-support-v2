"""Port for searching the web for current, external information."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    """One web search hit."""

    title: str
    url: str
    snippet: str
    retrieved_at: str


class WebSearchPort(Protocol):
    """Consumer-defined port for real-time web search."""

    def search(self, query: str, *, timeout_seconds: float) -> list[WebSearchResult]:
        """Search the web for `query` or raise `WebSearchUnavailableError`."""
        ...
