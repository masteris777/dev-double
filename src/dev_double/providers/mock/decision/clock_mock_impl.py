"""A deterministic clock: starts at ``start`` and advances ``step`` seconds per read."""

from __future__ import annotations

from dev_double.core.decision.i_clock import IClock


class ClockMockImpl(IClock):
    def __init__(self, start: float = 0.0, step: float = 0.0) -> None:
        self._t = start
        self._step = step

    def now(self) -> float:
        t = self._t
        self._t += self._step
        return t

    def advance(self, seconds: float) -> None:
        self._t += seconds
