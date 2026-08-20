"""Port for scoring corpus chunks against a query."""

from typing import Protocol

from getnet_support.domain.chat import Market
from getnet_support.domain.knowledge import RetrievedChunk


class RetrieverPort(Protocol):
    """Consumer-defined retrieval port (REQ-16: lexical and semantic parity)."""

    def search(
        self, query: str, *, market: Market | None = None, top_k: int = 20
    ) -> list[RetrievedChunk]:
        """Return corpus chunks ranked by relevance to `query`.

        `top_k` defaults generously above the committed corpus size (13
        chunks): the evidence gate (REQ-09) trusts `coverage`, not
        `score_retrieval`, to discriminate — a small `top_k` would let the
        weaker score signal exclude a high-coverage chunk before the gate
        ever sees it, defeating REQ-09's own rationale.
        """
        ...
