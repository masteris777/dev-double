"""Tool-call gating: allow, ask a human, or deny."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from .t_input import TInputValue
from .t_meta import TMeta
from .t_question import check_options


@dataclass(frozen=True)
class TToolCall:
    name: str
    arguments: Union[dict[str, Any], str] = field(default_factory=dict)


@dataclass(frozen=True)
class TGateRequest:
    tool_call: TToolCall
    context: Optional[TInputValue] = None  # what the user asked for, or the recent conversation
    policy: Optional[str] = None  # rules for tool use, in plain language
    outcomes: Optional[dict[str, str]] = None  # outcome -> when; None = DEFAULT_OUTCOMES

    def __post_init__(self) -> None:
        if self.outcomes is not None:
            check_options(self.outcomes, "outcomes")


@dataclass(frozen=True)
class TGateResponse:
    decision: str
    probabilities: dict[str, float]
    confidence: float
    meta: TMeta
