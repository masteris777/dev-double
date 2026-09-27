"""Input screening: injection, abuse, off-topic use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .defaults import DEFAULT_GUARD_THRESHOLD
from .t_input import TInputValue
from .t_meta import TMeta


@dataclass(frozen=True)
class TGuardRequest:
    input: TInputValue
    policies: Optional[dict[str, str]] = None  # policy -> violation; None = DEFAULT_POLICIES
    scope: Optional[str] = None  # what the assistant is for; adds an off_topic check
    threshold: float = DEFAULT_GUARD_THRESHOLD  # block when a violation probability reaches this

    def __post_init__(self) -> None:
        if self.policies is not None and not self.policies:
            raise ValueError("`policies` must not be empty.")
        if not 0 <= self.threshold <= 1:
            raise ValueError("`threshold` must be between 0 and 1.")


@dataclass(frozen=True)
class TGuardCheck:
    probability: float
    flagged: bool


@dataclass(frozen=True)
class TGuardResponse:
    allowed: bool
    flagged: list[str]
    checks: dict[str, TGuardCheck]
    meta: TMeta
