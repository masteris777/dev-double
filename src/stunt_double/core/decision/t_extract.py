"""Structured extraction: typed fields pulled out of one input.

Enum and boolean fields are answered as choice and binary questions (real
probabilities); string, number, and integer fields are generated as text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from .t_input import TInputValue
from .t_meta import TMeta
from .t_question import check_options

FIELD_TYPES = ("string", "number", "integer", "boolean", "enum")
TEXT_TYPES = ("string", "number", "integer")  # generated as text, then parsed
MAX_FIELDS = 30

# An extracted value; None when the input doesn't contain the field.
TValue = Optional[Union[str, float, int, bool]]


def _check_probability(p: Optional[float], what: str) -> None:
    if p is not None and not 0 <= p <= 1:
        raise ValueError(f"`{what}` must be between 0 and 1, got {p}.")


@dataclass(frozen=True)
class TExtractField:
    type: str  # one of FIELD_TYPES
    description: Optional[str] = None
    # enum only: option -> description, or a list of option names (each its own description).
    options: Optional[Union[dict[str, str], list[str]]] = None

    def __post_init__(self) -> None:
        if self.type not in FIELD_TYPES:
            raise ValueError(f"Field `type` must be one of {', '.join(FIELD_TYPES)}; got {self.type!r}.")
        if self.type != "enum":
            if self.options is not None:
                raise ValueError(f"Only enum fields take `options`, not {self.type} fields.")
            return
        if self.options is None:
            raise ValueError("Enum fields need `options`.")
        options = self.options
        if isinstance(options, list):
            if len(set(options)) != len(options):
                raise ValueError("Duplicate enum options.")
            options = {name: name for name in options}
        check_options(options)
        object.__setattr__(self, "options", dict(options))


@dataclass(frozen=True)
class TExtractRequest:
    input: TInputValue
    fields: dict[str, TExtractField]  # field name -> spec; answers keep this order

    def __post_init__(self) -> None:
        if not 1 <= len(self.fields) <= MAX_FIELDS:
            raise ValueError(f"`fields` must have 1 to {MAX_FIELDS} entries, got {len(self.fields)}.")
        for name in self.fields:
            if not name.strip():
                raise ValueError("Field names must not be empty.")


@dataclass(frozen=True)
class TFieldValue:
    value: TValue
    confidence: float
    probabilities: Optional[dict[str, float]] = None  # enum fields only
    probability: Optional[float] = None  # boolean fields only: probability of true

    def __post_init__(self) -> None:
        _check_probability(self.confidence, "confidence")
        _check_probability(self.probability, "probability")


EMPTY = TFieldValue(None, 0.0)


@dataclass(frozen=True)
class TExtractResponse:
    fields: dict[str, TFieldValue]
    meta: TMeta

    @property
    def values(self) -> dict[str, TValue]:
        return {name: f.value for name, f in self.fields.items()}
