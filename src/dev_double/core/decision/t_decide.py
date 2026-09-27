"""Generic decision: several typed questions about one input."""

from __future__ import annotations

from dataclasses import dataclass

from .t_answer import TAnswer
from .t_input import TInputValue
from .t_meta import TMeta
from .t_question import TQuestion


@dataclass(frozen=True)
class TDecideRequest:
    input: TInputValue
    questions: dict[str, TQuestion]

    def __post_init__(self) -> None:
        if not self.questions:
            raise ValueError("`questions` must not be empty.")


@dataclass(frozen=True)
class TDecideResponse:
    answers: dict[str, TAnswer]
    meta: TMeta
