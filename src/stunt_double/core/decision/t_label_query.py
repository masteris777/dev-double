"""The engine contract's data: a rendered single-label prompt in, a
distribution over those labels out."""

from __future__ import annotations

from dataclasses import dataclass, field

from .t_usage import TUsage


@dataclass
class TLabelQuery:
    """One question, rendered and ready for an engine."""

    system: str
    user: str
    labels: list[str]
    # Semantic parts, for engines that don't use the rendered prompt (e.g. mock).
    input_text: str
    question: str
    descriptions: list[str]


@dataclass
class TLabelResult:
    probabilities: dict[str, float]
    usage: TUsage = field(default_factory=TUsage)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TPosition:
    """One generated token and the top alternatives the model considered there."""

    token: str
    candidates: list[tuple[str, float]]  # (token, logprob)
