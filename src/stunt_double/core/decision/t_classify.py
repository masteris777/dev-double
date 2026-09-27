"""Labeling one item or a batch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .defaults import DEFAULT_CLASSIFY_QUESTION
from .t_input import TInputValue
from .t_meta import TMeta
from .t_question import check_options


@dataclass(frozen=True)
class TClassifyRequest:
    labels: dict[str, str]
    input: Optional[TInputValue] = None  # one item...
    inputs: Optional[list[TInputValue]] = None  # ...or many, labeled in parallel
    question: str = DEFAULT_CLASSIFY_QUESTION

    def __post_init__(self) -> None:
        check_options(self.labels, "labels")
        if (self.input is None) == (self.inputs is None):
            raise ValueError("Provide exactly one of `input` or `inputs`.")
        if self.inputs is not None and not self.inputs:
            raise ValueError("`inputs` must not be empty.")

    def items(self) -> list[TInputValue]:
        return self.inputs if self.inputs is not None else [self.input]  # type: ignore[list-item]


@dataclass(frozen=True)
class TClassification:
    label: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class TClassifyResponse:
    results: list[TClassification]
    meta: TMeta
