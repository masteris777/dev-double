"""LLM evals: score an output against a rubric."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .defaults import DEFAULT_JUDGE_CRITERIA
from .t_input import TInputValue
from .t_meta import TMeta
from .t_question import check_levels


@dataclass(frozen=True)
class TJudgeRequest:
    output: TInputValue  # the answer being judged
    input: Optional[TInputValue] = None  # the prompt or task that produced it
    reference: Optional[TInputValue] = None  # optional reference answer
    criteria: str = DEFAULT_JUDGE_CRITERIA
    levels: Optional[list[str]] = None  # ordered rubric, worst first; None = DEFAULT_RUBRIC

    def __post_init__(self) -> None:
        if self.levels is not None:
            check_levels(self.levels)


@dataclass(frozen=True)
class TJudgeResponse:
    score: float  # probability-weighted level, from 0 to len(levels) - 1
    normalized: float  # score / (len(levels) - 1), from 0 to 1
    level: int
    probabilities: dict[str, float]
    legend: dict[str, str]
    confidence: float
    meta: TMeta
