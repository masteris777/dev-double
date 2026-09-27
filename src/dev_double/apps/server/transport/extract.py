"""/v1/extract: typed fields pulled out of one input.

Invariants (field types, enum options, field counts) live in the core types;
the validators below construct them so a violation becomes a 422 that points
at the offending field.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_serializer, model_validator

from ....core.decision.t_extract import (
    MAX_FIELDS,
    TExtractField,
    TExtractRequest,
    TExtractResponse,
    TFieldValue,
)
from ....core.decision.t_question import MAX_OPTIONS
from .common import InputValue, Meta

Value = Optional[Union[bool, int, float, str]]


class ExtractField(BaseModel):
    type: Literal["string", "number", "integer", "boolean", "enum"] = Field(
        description="string, number, and integer values are generated; enum and boolean are scored."
    )
    description: Optional[str] = Field(None, description="What the field means and how to format it.")
    options: Optional[Union[list[str], dict[str, str]]] = Field(
        None,
        min_length=2,
        max_length=MAX_OPTIONS,
        description="Enum only: option names, or option -> description. The chosen name is returned.",
    )

    @model_validator(mode="after")
    def _core_invariants(self) -> "ExtractField":
        self.to_core()
        return self

    def to_core(self) -> TExtractField:
        options = list(self.options) if isinstance(self.options, list) else self.options
        return TExtractField(type=self.type, description=self.description, options=options)


class ExtractRequest(BaseModel):
    input: InputValue = Field(description="The content to extract from: text, an object, or a list.")
    fields: dict[str, ExtractField] = Field(
        min_length=1, max_length=MAX_FIELDS, description="Field name -> type and description."
    )

    @model_validator(mode="after")
    def _core_invariants(self) -> "ExtractRequest":
        self.to_core()
        return self

    def to_core(self) -> TExtractRequest:
        return TExtractRequest(input=self.input, fields={k: f.to_core() for k, f in self.fields.items()})


class FieldValue(BaseModel):
    value: Value = Field(None, description="The extracted value; null when the input doesn't contain it.")
    confidence: float
    probabilities: Optional[dict[str, float]] = Field(None, description="Enum fields only.")
    probability: Optional[float] = Field(None, description="Boolean fields only: probability of true.")

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler: Any) -> dict[str, Any]:
        # `value: null` is an answer; a missing probability is not, so omit only those.
        data = handler(self)
        rest = {k: v for k, v in data.items() if k != "value" and v is not None}
        return {"value": self.value, **rest}

    @classmethod
    def from_core(cls, t: TFieldValue) -> "FieldValue":
        return cls(value=t.value, confidence=t.confidence, probabilities=t.probabilities, probability=t.probability)


class ExtractResponse(BaseModel):
    values: dict[str, Value] = Field(description="Field name -> value, for direct use.")
    fields: dict[str, FieldValue] = Field(description="Field name -> value with its confidence.")
    meta: Meta

    @classmethod
    def from_core(cls, t: TExtractResponse) -> "ExtractResponse":
        return cls(
            values=dict(t.values),
            fields={k: FieldValue.from_core(f) for k, f in t.fields.items()},
            meta=Meta.from_core(t.meta),
        )
