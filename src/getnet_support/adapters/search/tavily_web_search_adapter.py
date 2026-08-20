"""Tavily-backed `WebSearchPort` implementation."""

import random
import time
from datetime import UTC, datetime
from typing import Any

import requests
from tavily import TavilyClient
from tavily.errors import TimeoutError as TavilyTimeoutError

from getnet_support.application.errors import WebSearchUnavailableError
from getnet_support.application.ports.web_search_port import WebSearchResult

_MAX_RESULTS = 3
_MAX_ATTEMPTS = 2
_BACKOFF_BASE_SECONDS = 0.2
_TRANSIENT_ERRORS = (
    TavilyTimeoutError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


class TavilyWebSearchAdapter:
    """Calls Tavily with an explicit timeout and one bounded transient retry."""

    def __init__(self, api_key: str) -> None:
        """Store the API key; a client is built per call to honor the timeout."""
        self._api_key = api_key

    def search(self, query: str, *, timeout_seconds: float) -> list[WebSearchResult]:
        """Search the web for `query`, retrying only transient failures."""
        client = TavilyClient(api_key=self._api_key)
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response: dict[str, Any] = client.search(
                    query=query, max_results=_MAX_RESULTS, timeout=timeout_seconds
                )
                return self._to_results(response)
            except _TRANSIENT_ERRORS as exc:
                last_error = exc
                if attempt + 1 < _MAX_ATTEMPTS:
                    # S311/B311: retry jitter, not a cryptographic use of random;
                    # permanent, not tech debt, no ticket needed.
                    jitter = random.uniform(0, 0.1)  # noqa: S311  # nosec B311
                    time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt) + jitter)
            except Exception as exc:
                raise WebSearchUnavailableError("Tavily rejected the request") from exc
        raise WebSearchUnavailableError("Tavily search timed out after retries") from last_error

    def _to_results(self, response: dict[str, Any]) -> list[WebSearchResult]:
        """Translate Tavily's raw response into port-owned result objects."""
        retrieved_at = datetime.now(UTC).date().isoformat()
        return [
            WebSearchResult(
                title=result.get("title", ""),
                url=result.get("url", ""),
                snippet=result.get("content", ""),
                retrieved_at=retrieved_at,
            )
            for result in response.get("results", [])
        ]
