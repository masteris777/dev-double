"""Answers to the three question types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class TBinaryAnswer:
    value: bool
    probability: float  # probability that the answer is yes
    confidence: float


@dataclass(frozen=True)
class TChoiceAnswer:
    value: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class TScaleAnswer:
    value: float  # probability-weighted level, from 0 to len(levels) - 1
    level: int  # the single most likely level
    probabilities: dict[str, float]
    legend: dict[str, str]
    confidence: float


TAnswer = Union[TBinaryAnswer, TChoiceAnswer, TScaleAnswer]
