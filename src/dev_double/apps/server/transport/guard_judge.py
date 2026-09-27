"""/v1/guard (binary questions per policy) and /v1/judge (one scale question)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from ....core.decision.defaults import DEFAULT_GUARD_THRESHOLD, DEFAULT_JUDGE_CRITERIA
from ....core.decision.t_guard import TGuardRequest, TGuardResponse
from ....core.decision.t_judge import TJudgeRequest, TJudgeResponse
from ....core.decision.t_question import MAX_LEVELS
from .common import InputValue, Meta


class GuardRequest(BaseModel):
    input: InputValue
    policies: Optional[dict[str, str]] = Field(
        None,
        min_length=1,
        description="Policy name -> description of a violation. Defaults to prompt_injection and abuse.",
    )
    scope: Optional[str] = Field(
        None, description="Optional: what the assistant is for. Adds an off_topic check."
    )
    threshold: float = Field(
        DEFAULT_GUARD_THRESHOLD, ge=0, le=1, description="Block when any violation probability reaches this."
    )

    @model_validator(mode="after")
    def _core_invariants(self) -> "GuardRequest":
        self.to_core()
        return self

    def to_core(self) -> TGuardRequest:
        return TGuardRequest(input=self.input, policies=self.policies, scope=self.scope, threshold=self.threshold)


class GuardCheck(BaseModel):
    probability: float
    flagged: bool


class GuardResponse(BaseModel):
    allowed: bool
    flagged: list[str]
    checks: dict[str, GuardCheck]
    meta: Meta

    @classmethod
    def from_core(cls, t: TGuardResponse) -> "GuardResponse":
        return cls(
            allowed=t.allowed,
            flagged=list(t.flagged),
            checks={k: GuardCheck(probability=c.probability, flagged=c.flagged) for k, c in t.checks.items()},
            meta=Meta.from_core(t.meta),
        )


class JudgeRequest(BaseModel):
    output: InputValue = Field(description="The answer being judged.")
    input: Optional[InputValue] = Field(None, description="The prompt or task that produced the output.")
    reference: Optional[InputValue] = Field(None, description="Optional reference answer.")
    criteria: str = DEFAULT_JUDGE_CRITERIA
    levels: Optional[list[str]] = Field(
        None,
        min_length=2,
        max_length=MAX_LEVELS,
        description="Ordered rubric levels, worst first. Defaults to a 5-level rubric (0-4).",
    )

    def to_core(self) -> TJudgeRequest:
        return TJudgeRequest(
            output=self.output, input=self.input, reference=self.reference, criteria=self.criteria, levels=self.levels
        )


class JudgeResponse(BaseModel):
    score: float = Field(description="Probability-weighted level, from 0 to len(levels) - 1.")
    normalized: float = Field(description="score / (len(levels) - 1), from 0 to 1.")
    level: int
    probabilities: dict[str, float]
    legend: dict[str, str]
    confidence: float
    meta: Meta

    @classmethod
    def from_core(cls, t: TJudgeResponse) -> "JudgeResponse":
        return cls(
            score=t.score,
            normalized=t.normalized,
            level=t.level,
            probabilities=t.probabilities,
            legend=t.legend,
            confidence=t.confidence,
            meta=Meta.from_core(t.meta),
        )
