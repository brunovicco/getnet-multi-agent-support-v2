"""Small bounded-retry helper for transient external-call failures.

Shared by the Gemini and Tavily adapters: one retry, exponential backoff, jitter — never used for
non-idempotent, state-changing calls.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable

MAX_ATTEMPTS = 2
BASE_DELAY_SECONDS = 0.25
_jitter = random.SystemRandom()


async def with_bounded_retry[T](
    operation: Callable[[], Awaitable[T]],
    *,
    is_transient: Callable[[Exception], bool],
) -> T:
    """Run `operation`, retrying once with jittered backoff on a transient failure."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return await operation()
        except Exception as exc:
            if not is_transient(exc) or attempt == MAX_ATTEMPTS - 1:
                raise
            delay = BASE_DELAY_SECONDS * (2**attempt) + _jitter.uniform(0, 0.1)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable: loop always returns or raises")
