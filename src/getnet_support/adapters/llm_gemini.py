"""Gemini LLM adapter: thin REST client behind LLMPort.

Calls the Gemini REST API directly via httpx instead of the vendor SDK, keeping the runtime
dependency surface minimal. Network and API failures are mapped to LLMGenerationError so
application code never depends on httpx or on the Gemini response shape.

The API key is sent as the `x-goog-api-key` header, never as a `?key=` query parameter: httpx's
own request logger prints the full request URL at INFO level, so a query-string key would land in
every log stream (including this project's default JSON stdout logs) even without DEBUG logging.
"""

import httpx

from getnet_support.adapters.retry import with_bounded_retry
from getnet_support.application.ports import LLMGenerationError, LLMPort
from getnet_support.domain.models import Locale

DEFAULT_MODEL = "gemini-2.5-flash"
_TIMEOUT_SECONDS = 20.0
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, httpx.TimeoutException | httpx.ConnectError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


class GeminiLLMAdapter(LLMPort):
    """Calls the Gemini `generateContent` REST endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Bind the API key and model; `transport` is a test-only injection seam."""
        self._api_key = api_key
        self._model = model
        self._transport = transport

    async def generate(self, *, system_prompt: str, user_prompt: str, locale: Locale) -> str:
        """Call Gemini and return the generated text, or raise LLMGenerationError."""
        del locale  # the instruction to answer in the target language is in `system_prompt`
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
        }

        async def _call() -> httpx.Response:
            async with httpx.AsyncClient(
                timeout=_TIMEOUT_SECONDS, transport=self._transport
            ) as client:
                response = await client.post(
                    f"{_BASE_URL}/{self._model}:generateContent",
                    headers={"x-goog-api-key": self._api_key},
                    json=payload,
                )
                response.raise_for_status()
                return response

        try:
            response = await with_bounded_retry(_call, is_transient=_is_transient)
        except httpx.HTTPError as exc:
            raise LLMGenerationError(f"Gemini request failed: {exc}") from exc

        try:
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return str(text).strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMGenerationError("Gemini response missing generated text") from exc
