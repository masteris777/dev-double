"""The process's monotonic high-resolution clock."""

from __future__ import annotations

import time

from dev_double.core.decision.i_clock import IClock


class ClockStdImpl(IClock):
    def now(self) -> float:
        return time.perf_counter()
