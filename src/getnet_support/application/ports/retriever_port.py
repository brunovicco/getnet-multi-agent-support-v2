"""Port for scoring corpus chunks against a query."""

from typing import Protocol

from getnet_support.domain.chat import Market
from getnet_support.domain.knowledge import RetrievedChunk


class RetrieverPort(Protocol):
    """Consumer-defined retrieval port (REQ-16: lexical and semantic parity)."""

    def search(
        self, query: str, *, market: Market | None = None, top_k: int = 3
    ) -> list[RetrievedChunk]:
        """Return corpus chunks ranked by relevance to `query`."""
        ...
