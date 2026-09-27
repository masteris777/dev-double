"""Collects usage, warnings, and latency across the model calls of one request."""

from __future__ import annotations

from .i_clock import IClock
from .t_meta import TMeta
from .t_usage import TUsage


class Tracker:
    def __init__(self, clock: IClock) -> None:
        self._clock = clock
        self.usage = TUsage()
        self.warnings: list[str] = []
        self.started = clock.now()

    def warn(self, message: str, label: str = "") -> None:
        self.warnings.append(f"{label}: {message}" if label else message)

    def meta(self, engine: str, model: str) -> TMeta:
        return TMeta(
            engine=engine,
            model=model,
            latency_ms=round((self._clock.now() - self.started) * 1000),
            usage=self.usage,
            warnings=list(dict.fromkeys(self.warnings)),
        )
