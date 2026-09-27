"""Response metadata shared by every use case."""

from __future__ import annotations

from dataclasses import dataclass, field

from .t_usage import TUsage


@dataclass(frozen=True)
class TMeta:
    engine: str
    model: str
    latency_ms: int
    usage: TUsage
    # Where this answer may be less faithful than a real decision model.
    warnings: list[str] = field(default_factory=list)
