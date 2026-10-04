"""/v1/systemone: typed questions about one shared state, answered together.

The contract follows Ollama's /v1/systemone OpenAPI spec (MIT License,
Copyright (c) Ollama), plus one extra top-level field, ``meta``.

The wire mapping lives in ``core.decision.systemone_wire`` (shared with the
System 1 decider); the models below describe the schema for OpenAPI and
construct the core request, so every contract violation becomes a validation
error that the route reports as a 400.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....core.decision import systemone_wire as wire
from ....core.decision.t_decide import TDecideRequest, TDecideResponse
from ....core.decision.t_question import MAX_LEVELS, MAX_OPTIONS
from .common import Meta, Usage

Content = Union[str, dict[str, Any], list[Any]]
CONTENT_DESCRIPTION = "A non-blank string, or an object or array serialized as JSON text."


class SystemOneChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Content = Field(description=CONTENT_DESCRIPTION)
    criteria: dict[str, Optional[str]] = Field(
        description="Option key -> description; a null description uses the key itself.",
        json_schema_extra={"minProperties": 2, "maxProperties": MAX_OPTIONS},
    )


class SystemOneNoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true: Optional[str] = Field(None, description="What counts as yes.")
    false: Optional[str] = Field(None, description="What counts as no.")


class SystemOneNoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Content = Field(description=CONTENT_DESCRIPTION)
    criteria: Optional[SystemOneNoulCriteria] = Field(
        None, description="Optional descriptions for the two outcomes, under the keys true and false."
    )


class SystemOneScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Content = Field(description=CONTENT_DESCRIPTION)
    criteria: list[str] = Field(
        description="Descriptions from the lowest score (index 0) to the highest.",
        json_schema_extra={"minItems": 2, "maxItems": MAX_LEVELS},
    )


SystemOneQuestion = Annotated[
    Union[SystemOneChoiceQuestion, SystemOneNoulQuestion, SystemOneScoreQuestion],
    Field(discriminator="type"),
]


class SystemOneRequest(BaseModel):
    model: str = Field(
        description="Echoed in the response. The configured model answers unless the server "
        "runs with --honor-request-model."
    )
    state: Content = Field(description=CONTENT_DESCRIPTION)
    images: Optional[list[str]] = Field(
        None,
        description="Base64-encoded PNG, JPEG, or WebP images shared by all questions, in request order. "
        "No URLs or data URLs. Supported by the systemone engine, and by the openai engine "
        "when it has a --vision-model.",
    )
    questions: dict[str, SystemOneQuestion] = Field(
        description="Named questions about the shared state.",
        json_schema_extra={"minProperties": 1, "maxProperties": wire.MAX_QUESTIONS},
    )
    keep_alive: Optional[Union[str, float]] = Field(None, description="Accepted and ignored.")

    @model_validator(mode="after")
    def _core_invariants(self) -> "SystemOneRequest":
        if not self.model.strip():
            raise ValueError("`model` must not be blank.")
        self.to_core()
        return self

    def to_core(self) -> TDecideRequest:
        questions = wire.questions_from_wire(
            {name: q.model_dump(mode="json", exclude_none=False) for name, q in self.questions.items()}
        )
        return TDecideRequest(
            input=wire.state_from_wire(self.state),
            questions=questions,
            images=wire.images_from_wire(self.images),
        )


class SystemOneChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str = Field(description="Option key with the highest probability.")
    probabilities: dict[str, float]
    confidence: float


class SystemOneNoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float = Field(description="Probability of true. A number, not a Boolean.")


class SystemOneScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float = Field(description="Probability-weighted average of the zero-based criterion indices.")
    legend: dict[str, str] = Field(description="Zero-based indices as string keys -> criterion descriptions.")
    probabilities: dict[str, float]
    confidence: float


SystemOneAnswer = Annotated[
    Union[SystemOneChoiceAnswer, SystemOneNoulAnswer, SystemOneScoreAnswer],
    Field(discriminator="type"),
]


class SystemOneResponse(BaseModel):
    model: str = Field(description="Model name from the request.")
    answers: dict[str, SystemOneAnswer] = Field(description="Answers keyed by the question names in the request.")
    usage: Usage
    meta: Meta = Field(description="dev-double's own metadata (not part of the System One contract).")

    @classmethod
    def from_core(cls, model: str, t: TDecideResponse) -> "SystemOneResponse":
        return cls(**wire.response_to_wire(model, t.answers, t.meta.usage), meta=Meta.from_core(t.meta))


class ErrorResponse(BaseModel):
    error: str = Field(description="Error message describing what went wrong.")
