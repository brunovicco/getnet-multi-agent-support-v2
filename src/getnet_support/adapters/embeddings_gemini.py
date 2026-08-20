"""Gemini embedding adapter: thin REST client behind EmbeddingPort.

Calls the Gemini `batchEmbedContents` REST endpoint directly via httpx, keeping the runtime
dependency surface minimal — same pattern as `llm_gemini.py`.

The API key is sent as the `x-goog-api-key` header, never as a `?key=` query parameter: httpx's
own request logger prints the full request URL at INFO level, so a query-string key would land in
every log stream (including this project's default JSON stdout logs) even without DEBUG logging.
"""

import httpx

from getnet_support.adapters.retry import with_bounded_retry
from getnet_support.application.ports import EmbeddingGenerationError, EmbeddingPort

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
_TIMEOUT_SECONDS = 20.0
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, httpx.TimeoutException | httpx.ConnectError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


class GeminiEmbeddingAdapter(EmbeddingPort):
    """Calls the Gemini `batchEmbedContents` REST endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_EMBEDDING_MODEL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Bind the API key and model; `transport` is a test-only injection seam."""
        self._api_key = api_key
        self._model = model
        self._transport = transport

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        """Embed every text in one batched call, or raise EmbeddingGenerationError."""
        if not texts:
            return ()
        model_name = f"models/{self._model}"
        payload = {
            "requests": [
                {"model": model_name, "content": {"parts": [{"text": text}]}} for text in texts
            ]
        }

        async def _call() -> httpx.Response:
            async with httpx.AsyncClient(
                timeout=_TIMEOUT_SECONDS, transport=self._transport
            ) as client:
                response = await client.post(
                    f"{_BASE_URL}/{self._model}:batchEmbedContents",
                    headers={"x-goog-api-key": self._api_key},
                    json=payload,
                )
                response.raise_for_status()
                return response

        try:
            response = await with_bounded_retry(_call, is_transient=_is_transient)
        except httpx.HTTPError as exc:
            raise EmbeddingGenerationError(f"Gemini embedding request failed: {exc}") from exc

        try:
            data = response.json()
            return tuple(
                tuple(float(value) for value in item["values"]) for item in data["embeddings"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingGenerationError("Gemini embedding response was malformed") from exc
