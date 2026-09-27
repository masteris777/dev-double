"""/v1/route, /v1/gate, /v1/classify: the use cases answered by one choice question."""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from ....core.decision.defaults import DEFAULT_CLASSIFY_QUESTION
from ....core.decision.t_classify import TClassifyRequest, TClassifyResponse
from ....core.decision.t_gate import TGateRequest, TGateResponse, TToolCall
from ....core.decision.t_question import MAX_OPTIONS, check_option_keys
from ....core.decision.t_route import TRouteRequest, TRouteResponse
from .common import InputValue, Meta


class _CoreValidated(BaseModel):
    """Constructs the core type after field validation, so core invariants give a 422."""

    @model_validator(mode="after")
    def _core_invariants(self):  # type: ignore[no-untyped-def]
        self.to_core()  # type: ignore[attr-defined]
        return self


def _distinct_keys(v: Optional[dict[str, str]]) -> Optional[dict[str, str]]:
    # Checked per field too, so the 422 points at the offending field.
    if v is not None:
        check_option_keys(v)
    return v


class RouteRequest(_CoreValidated):
    input: InputValue = Field(description="The prompt or task to route.")
    routes: Optional[dict[str, str]] = Field(
        None,
        min_length=2,
        max_length=MAX_OPTIONS,
        description="Route name -> when to use it. Defaults to small / medium / large model tiers.",
    )

    _routes = field_validator("routes")(_distinct_keys)

    def to_core(self) -> TRouteRequest:
        return TRouteRequest(input=self.input, routes=self.routes)


class RouteResponse(BaseModel):
    route: str
    probabilities: dict[str, float]
    confidence: float
    meta: Meta

    @classmethod
    def from_core(cls, t: TRouteResponse) -> "RouteResponse":
        return cls(route=t.route, probabilities=t.probabilities, confidence=t.confidence, meta=Meta.from_core(t.meta))


class ToolCall(BaseModel):
    name: str
    arguments: Union[dict[str, Any], str] = Field(default_factory=dict)


class GateRequest(_CoreValidated):
    tool_call: ToolCall
    context: Optional[InputValue] = Field(None, description="What the user asked for, or the recent conversation.")
    policy: Optional[str] = Field(None, description="Your rules for tool use, in plain language.")
    outcomes: Optional[dict[str, str]] = Field(
        None,
        min_length=2,
        max_length=MAX_OPTIONS,
        description="Outcome -> when it applies. Defaults to allow / ask / deny.",
    )

    _outcomes = field_validator("outcomes")(_distinct_keys)

    def to_core(self) -> TGateRequest:
        call = TToolCall(name=self.tool_call.name, arguments=self.tool_call.arguments)
        return TGateRequest(tool_call=call, context=self.context, policy=self.policy, outcomes=self.outcomes)


class GateResponse(BaseModel):
    decision: str
    probabilities: dict[str, float]
    confidence: float
    meta: Meta

    @classmethod
    def from_core(cls, t: TGateResponse) -> "GateResponse":
        return cls(decision=t.decision, probabilities=t.probabilities, confidence=t.confidence, meta=Meta.from_core(t.meta))


class ClassifyRequest(_CoreValidated):
    input: Optional[InputValue] = Field(None, description="One item. Use `inputs` for a batch.")
    inputs: Optional[list[InputValue]] = Field(None, description="Many items, labeled in parallel.")
    labels: dict[str, str] = Field(min_length=2, max_length=MAX_OPTIONS)
    question: str = DEFAULT_CLASSIFY_QUESTION

    _labels = field_validator("labels")(_distinct_keys)

    def to_core(self) -> TClassifyRequest:
        return TClassifyRequest(labels=self.labels, input=self.input, inputs=self.inputs, question=self.question)


class Classification(BaseModel):
    label: str
    probabilities: dict[str, float]
    confidence: float


class ClassifyResponse(BaseModel):
    results: list[Classification]
    meta: Meta

    @classmethod
    def from_core(cls, t: TClassifyResponse) -> "ClassifyResponse":
        results = [
            Classification(label=r.label, probabilities=r.probabilities, confidence=r.confidence)
            for r in t.results
        ]
        return cls(results=results, meta=Meta.from_core(t.meta))
