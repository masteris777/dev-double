"""Request and response models for every endpoint.

The core is one generic endpoint (`/v1/decide`) with three question types:

- ``binary``: is this statement true? -> probability of "yes"
- ``choice``: which of these unordered options? -> option + distribution
- ``scale``:  which of these ordered levels? -> expected level + distribution

The use-case endpoints (route, guard, gate, classify, judge, rerank) are thin
wrappers that build these questions for you.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# Probabilities are read from the model's top 20 candidate tokens (the most
# OpenAI-compatible servers return), so more options than that can't all be seen.
MAX_OPTIONS = 20
MAX_LEVELS = 10

InputValue = Union[str, dict[str, Any], list[Any]]


def check_option_keys(options: dict[str, str]) -> None:
    """The model answers with an option key, compared ignoring case and
    punctuation, so keys must stay distinct under that comparison."""
    seen: dict[str, str] = {}
    for key in options:
        norm = "".join(ch for ch in key.lower() if ch.isalnum())
        if not norm:
            raise ValueError(f"Option key {key!r} must contain a letter or digit.")
        if norm in seen:
            raise ValueError(f"Option keys {seen[norm]!r} and {key!r} are too similar; rename one.")
        seen[norm] = key


# --------------------------------------------------------------------------
# Core: /v1/decide
# --------------------------------------------------------------------------


class BinaryQuestion(BaseModel):
    type: Literal["binary"]
    question: str = Field(description="A yes/no question or a statement to verify.")
    yes: str | None = Field(None, description="Optional: what counts as yes.")
    no: str | None = Field(None, description="Optional: what counts as no.")


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    question: str
    options: dict[str, str] = Field(
        min_length=2,
        max_length=MAX_OPTIONS,
        description="Option key -> description. Keys are returned as the answer.",
    )

    @model_validator(mode="after")
    def _distinct_keys(self) -> "ChoiceQuestion":
        check_option_keys(self.options)
        return self


class ScaleQuestion(BaseModel):
    type: Literal["scale"]
    question: str
    levels: list[str] = Field(
        min_length=2,
        max_length=MAX_LEVELS,
        description="Ordered level descriptions, lowest first. Level i is returned as i.",
    )


Question = Annotated[
    Union[BinaryQuestion, ChoiceQuestion, ScaleQuestion], Field(discriminator="type")
]


class DecideRequest(BaseModel):
    input: InputValue = Field(description="The content to decide about: text, an object, or a list.")
    questions: dict[str, Question] = Field(min_length=1)
    model: str | None = Field(None, description="Ignored; the server's configured model is used.")


class BinaryAnswer(BaseModel):
    type: Literal["binary"] = "binary"
    value: bool
    probability: float = Field(description="Probability that the answer is yes.")
    confidence: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    value: str
    probabilities: dict[str, float]
    confidence: float


class ScaleAnswer(BaseModel):
    type: Literal["scale"] = "scale"
    value: float = Field(description="Probability-weighted level, from 0 to len(levels) - 1.")
    level: int = Field(description="The single most likely level.")
    probabilities: dict[str, float]
    legend: dict[str, str]
    confidence: float


Answer = Annotated[Union[BinaryAnswer, ChoiceAnswer, ScaleAnswer], Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


class Meta(BaseModel):
    engine: str
    model: str
    latency_ms: int
    usage: Usage
    warnings: list[str] = Field(
        default_factory=list,
        description="Where this answer may be less faithful than a real decision model.",
    )


class DecideResponse(BaseModel):
    answers: dict[str, Answer]
    meta: Meta


# --------------------------------------------------------------------------
# Use cases
# --------------------------------------------------------------------------


class RouteRequest(BaseModel):
    input: InputValue = Field(description="The prompt or task to route.")
    routes: dict[str, str] | None = Field(
        None,
        min_length=2,
        max_length=MAX_OPTIONS,
        description="Route name -> when to use it. Defaults to small / medium / large model tiers.",
    )

    @field_validator("routes")
    @classmethod
    def _distinct_routes(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        if v is not None:
            check_option_keys(v)
        return v


class RouteResponse(BaseModel):
    route: str
    probabilities: dict[str, float]
    confidence: float
    meta: Meta


class GuardRequest(BaseModel):
    input: InputValue
    policies: dict[str, str] | None = Field(
        None,
        min_length=1,
        description="Policy name -> description of a violation. Defaults to prompt_injection and abuse.",
    )
    scope: str | None = Field(
        None, description="Optional: what the assistant is for. Adds an off_topic check."
    )
    threshold: float = Field(0.5, ge=0, le=1, description="Block when any violation probability reaches this.")


class GuardCheck(BaseModel):
    probability: float
    flagged: bool


class GuardResponse(BaseModel):
    allowed: bool
    flagged: list[str]
    checks: dict[str, GuardCheck]
    meta: Meta


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] | str = Field(default_factory=dict)


class GateRequest(BaseModel):
    tool_call: ToolCall
    context: InputValue | None = Field(None, description="What the user asked for, or the recent conversation.")
    policy: str | None = Field(None, description="Your rules for tool use, in plain language.")
    outcomes: dict[str, str] | None = Field(
        None,
        min_length=2,
        max_length=MAX_OPTIONS,
        description="Outcome -> when it applies. Defaults to allow / ask / deny.",
    )

    @field_validator("outcomes")
    @classmethod
    def _distinct_outcomes(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        if v is not None:
            check_option_keys(v)
        return v


class GateResponse(BaseModel):
    decision: str
    probabilities: dict[str, float]
    confidence: float
    meta: Meta


class ClassifyRequest(BaseModel):
    input: InputValue | None = Field(None, description="One item. Use `inputs` for a batch.")
    inputs: list[InputValue] | None = Field(None, description="Many items, labeled in parallel.")
    labels: dict[str, str] = Field(min_length=2, max_length=MAX_OPTIONS)
    question: str = "Which label best describes the input?"

    @field_validator("labels")
    @classmethod
    def _distinct_labels(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        if v is not None:
            check_option_keys(v)
        return v

    @model_validator(mode="after")
    def _one_of_input_or_inputs(self) -> "ClassifyRequest":
        if (self.input is None) == (self.inputs is None):
            raise ValueError("Provide exactly one of `input` or `inputs`.")
        if self.inputs is not None and not self.inputs:
            raise ValueError("`inputs` must not be empty.")
        return self

    def items(self) -> list[InputValue]:
        return self.inputs if self.inputs is not None else [self.input]  # type: ignore[list-item]


class Classification(BaseModel):
    label: str
    probabilities: dict[str, float]
    confidence: float


class ClassifyResponse(BaseModel):
    results: list[Classification]
    meta: Meta


class JudgeRequest(BaseModel):
    output: InputValue = Field(description="The answer being judged.")
    input: InputValue | None = Field(None, description="The prompt or task that produced the output.")
    reference: InputValue | None = Field(None, description="Optional reference answer.")
    criteria: str = "Overall quality: correct, complete, relevant, and clearly written."
    levels: list[str] | None = Field(
        None,
        min_length=2,
        max_length=MAX_LEVELS,
        description="Ordered rubric levels, worst first. Defaults to a 5-level rubric (0-4).",
    )


class JudgeResponse(BaseModel):
    score: float = Field(description="Probability-weighted level, from 0 to len(levels) - 1.")
    normalized: float = Field(description="score / (len(levels) - 1), from 0 to 1.")
    level: int
    probabilities: dict[str, float]
    legend: dict[str, str]
    confidence: float
    meta: Meta


# --------------------------------------------------------------------------
# Rerank: Cohere-style wire format, the de facto standard shared by
# Hugging Face TEI, Infinity, Jina, and vLLM.
# --------------------------------------------------------------------------


class RerankDocument(BaseModel):
    text: str


class RerankRequest(BaseModel):
    query: str
    documents: list[Union[str, RerankDocument]] = Field(min_length=1)
    top_n: int | None = Field(None, ge=1)
    return_documents: bool = False
    model: str | None = Field(None, description="Ignored; the server's configured model is used.")

    def texts(self) -> list[str]:
        return [d if isinstance(d, str) else d.text for d in self.documents]


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: RerankDocument | None = None


class RerankResponse(BaseModel):
    id: str
    results: list[RerankResult]
    meta: Meta
