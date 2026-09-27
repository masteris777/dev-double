"""timed(label, fn): run an integration call and record its latency.

Recorded timings are printed in the pytest terminal summary (see tests/conftest.py).
"""

from __future__ import annotations

import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")

TIMINGS: list[str] = []


async def timed(label: str, fn: Callable[[], Awaitable[T]]) -> T:
    started = time.perf_counter()
    try:
        return await fn()
    finally:
        line = f"[timing] {label}: {(time.perf_counter() - started) * 1000:.0f}ms"
        TIMINGS.append(line)
        print(line)
