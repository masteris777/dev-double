"""Time port."""

from __future__ import annotations

from typing import Protocol


class IClock(Protocol):
    def now(self) -> float:
        """Monotonic seconds; only differences are meaningful."""
        ...
