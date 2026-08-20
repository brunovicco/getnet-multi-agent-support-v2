"""Consumer-defined ports implemented by adapters.

Application code depends on these Protocols, never on a concrete SDK.
"""

from typing import Protocol

from getnet_support.domain.models import (
    CustomerProfile,
    Locale,
    Market,
    RetrievedChunk,
    TerminalStatus,
    Transaction,
    WebSearchResult,
)


class CustomerNotFoundError(Exception):
    """Raised when a customer-scoped tool has no data for the given user_id."""


class LLMGenerationError(Exception):
    """Raised when the LLM provider fails to generate a response."""


class WebSearchUnavailableError(Exception):
    """Raised when the web search provider call fails or is not configured."""


class EmbeddingGenerationError(Exception):
    """Raised when the embedding provider fails to embed one or more texts."""


class LLMPort(Protocol):
    """Port for grounded text generation, implemented by one provider adapter."""

    async def generate(self, *, system_prompt: str, user_prompt: str, locale: Locale) -> str:
        """Generate a grounded response, or raise LLMGenerationError on provider failure."""
        ...


class WebSearchPort(Protocol):
    """Port for current/external information lookup."""

    def is_configured(self) -> bool:
        """Return whether the provider has credentials configured."""
        ...

    async def search(self, query: str) -> tuple[WebSearchResult, ...]:
        """Return web search results, or raise WebSearchUnavailableError if the call fails."""
        ...


class KnowledgeRetrieverPort(Protocol):
    """Port for RAG retrieval over the persisted corpus (lexical or semantic)."""

    async def retrieve(
        self, query: str, *, market: Market, top_k: int = 3
    ) -> tuple[RetrievedChunk, ...]:
        """Return the top matching chunks scoped to market, best score first."""
        ...


class EmbeddingPort(Protocol):
    """Port for turning text into embedding vectors, implemented by one provider adapter."""

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        """Return one embedding vector per input text, in the same order.

        Raises EmbeddingGenerationError on provider failure.
        """
        ...


class CustomerDataPort(Protocol):
    """Port for customer-scoped tools backed by CRM/settlement/terminal systems."""

    def get_customer_profile(self, user_id: str) -> CustomerProfile:
        """Return the customer profile or raise CustomerNotFoundError."""
        ...

    def get_recent_transactions(self, user_id: str) -> tuple[Transaction, ...]:
        """Return recent settlement transactions or raise CustomerNotFoundError."""
        ...

    def get_terminal_status(self, user_id: str) -> TerminalStatus:
        """Return terminal connectivity status or raise CustomerNotFoundError."""
        ...
