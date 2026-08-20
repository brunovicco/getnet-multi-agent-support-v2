"""Tavily web search adapter: thin REST client behind WebSearchPort.

Used only for current/external questions (weather, exchange rates, news) that the local RAG
corpus cannot answer. Never used to look up customer data. Only the free-text question is ever
sent to Tavily — never the user_id or any customer-scoped data.
"""

import httpx

from getnet_support.adapters.retry import with_bounded_retry
from getnet_support.application.ports import WebSearchPort, WebSearchUnavailableError
from getnet_support.domain.models import WebSearchResult

_TIMEOUT_SECONDS = 10.0
_SEARCH_URL = "https://api.tavily.com/search"


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, httpx.TimeoutException | httpx.ConnectError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


class TavilyWebSearchAdapter(WebSearchPort):
    """Calls the Tavily `/search` REST endpoint."""

    def __init__(
        self, api_key: str | None, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        """Bind the API key; `transport` is a test-only injection seam."""
        self._api_key = api_key
        self._transport = transport

    def is_configured(self) -> bool:
        """Return whether TAVILY_API_KEY was provided."""
        return bool(self._api_key)

    async def search(self, query: str) -> tuple[WebSearchResult, ...]:
        """Return web search results, or raise WebSearchUnavailableError if the call fails."""
        if not self._api_key:
            raise WebSearchUnavailableError("Tavily API key is not configured")

        payload = {"api_key": self._api_key, "query": query, "max_results": 3}

        async def _call() -> httpx.Response:
            async with httpx.AsyncClient(
                timeout=_TIMEOUT_SECONDS, transport=self._transport
            ) as client:
                response = await client.post(_SEARCH_URL, json=payload)
                response.raise_for_status()
                return response

        try:
            response = await with_bounded_retry(_call, is_transient=_is_transient)
        except httpx.HTTPError as exc:
            raise WebSearchUnavailableError(f"Tavily request failed: {exc}") from exc

        try:
            data = response.json()
            return tuple(
                WebSearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("content", "")),
                )
                for item in data.get("results", [])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WebSearchUnavailableError("Tavily response was malformed") from exc
