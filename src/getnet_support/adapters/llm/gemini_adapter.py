"""Gemini-backed `LLMPort` implementation."""

import random
import time

from google import genai
from google.genai import errors, types

from getnet_support.application.errors import LLMUnavailableError

_MODEL = "gemini-2.5-flash"
_MAX_ATTEMPTS = 2
_BACKOFF_BASE_SECONDS = 0.2


class GeminiAdapter:
    """Calls Gemini with an explicit timeout and one bounded transient retry."""

    def __init__(self, api_key: str) -> None:
        """Store the API key; a client is built per call to honor the timeout."""
        self._api_key = api_key

    def generate(self, *, prompt: str, timeout_seconds: float) -> str:
        """Generate text for `prompt`, retrying only transient server errors."""
        client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = client.models.generate_content(model=_MODEL, contents=prompt)
                return (response.text or "").strip()
            except errors.ServerError as exc:
                last_error = exc
                if attempt + 1 < _MAX_ATTEMPTS:
                    # S311/B311: retry jitter, not a cryptographic use of random;
                    # permanent, not tech debt, no ticket needed.
                    jitter = random.uniform(0, 0.1)  # noqa: S311  # nosec B311
                    time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt) + jitter)
            except errors.ClientError as exc:
                raise LLMUnavailableError("Gemini rejected the request") from exc
        raise LLMUnavailableError("Gemini did not respond after retries") from last_error
