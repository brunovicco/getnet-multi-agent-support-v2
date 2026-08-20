"""Regression test for `_run_sync`, the async build-step bridge in the composition root.

Covers the real bug this fixes: `uvicorn --reload` (and multi-worker mode) re-imports
`entrypoints/http.py` inside a subprocess whose event loop is already running by the time the
module-level `app = create_app()` executes, so a plain `asyncio.run()` there raises
`RuntimeError: asyncio.run() cannot be called from a running event loop`.
"""

import asyncio

from getnet_support.entrypoints.http import _run_sync


async def _return_42() -> int:
    return 42


def test_run_sync_works_with_no_running_loop() -> None:
    assert _run_sync(_return_42()) == 42


def test_run_sync_works_inside_a_running_loop() -> None:
    async def _call_from_within_loop() -> int:
        return _run_sync(_return_42())

    assert asyncio.run(_call_from_within_loop()) == 42
