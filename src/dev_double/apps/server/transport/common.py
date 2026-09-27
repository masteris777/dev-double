"""Shared transport pieces: input values and response metadata."""

from __future__ import annotations

from typing import Any, Union

from pydantic import BaseModel, Field

from ....core.decision.t_meta import TMeta

InputValue = Union[str, dict[str, Any], list[Any]]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class Meta(BaseModel):
    engine: str
    model: str
    latency_ms: int
    usage: Usage
    warnings: list[str] = Field(
        default_factory=list,
        description="Where this answer may be less faithful than a real decision model.",
    )

    @classmethod
    def from_core(cls, t: TMeta) -> "Meta":
        return cls(
            engine=t.engine,
            model=t.model,
            latency_ms=t.latency_ms,
            usage=Usage(input_tokens=t.usage.input_tokens, output_tokens=t.usage.output_tokens),
            warnings=list(t.warnings),
        )
