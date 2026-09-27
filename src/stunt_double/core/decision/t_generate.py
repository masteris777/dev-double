"""The generator contract's data: a rendered prompt in, a short text and how
sure the model was of it out."""

from __future__ import annotations

from dataclasses import dataclass, field

from .t_usage import TUsage


@dataclass(frozen=True)
class TGenerateQuery:
    system: str
    user: str
    max_tokens: int
    json: bool = False  # ask for one JSON object (e.g. response_format json_object)
    # Semantic parts, for engines that don't use the rendered prompt (e.g. mock).
    input_text: str = ""
    fields: tuple[str, ...] = ()  # the field names being extracted

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("`max_tokens` must be at least 1.")


@dataclass
class TGenerateResult:
    text: str
    confidence: float  # 0..1 for the whole text, e.g. exp(mean token logprob)
    usage: TUsage = field(default_factory=TUsage)
    warnings: list[str] = field(default_factory=list)
    # The generated tokens as (text, logprob), in order; empty if the server returns none.
    tokens: list[tuple[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"`confidence` must be between 0 and 1, got {self.confidence}.")
