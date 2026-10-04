"""/v1/decide: three question types, answered about one input.

Invariants (option counts, distinct keys) live in the core types; the
validators below construct them so a violation becomes a 422.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from ....core.decision.t_answer import TAnswer, TBinaryAnswer, TChoiceAnswer
from ....core.decision.t_decide import TDecideRequest, TDecideResponse
from ....core.decision.t_question import (
    MAX_LEVELS,
    MAX_OPTIONS,
    TBinaryQuestion,
    TChoiceQuestion,
    TQuestion,
    TScaleQuestion,
)
from .common import InputValue, Meta


class BinaryQuestion(BaseModel):
    type: Literal["binary"]
    question: str = Field(description="A yes/no question or a statement to verify.")
    yes: Optional[str] = Field(None, description="Optional: what counts as yes.")
    no: Optional[str] = Field(None, description="Optional: what counts as no.")

    def to_core(self) -> TBinaryQuestion:
        return TBinaryQuestion(question=self.question, yes=self.yes, no=self.no)


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    question: str
    options: dict[str, str] = Field(
        min_length=2,
        max_length=MAX_OPTIONS,
        description="Option key -> description. Keys are returned as the answer.",
    )

    @model_validator(mode="after")
    def _core_invariants(self) -> "ChoiceQuestion":
        self.to_core()
        return self

    def to_core(self) -> TChoiceQuestion:
        return TChoiceQuestion(question=self.question, options=dict(self.options))


class ScaleQuestion(BaseModel):
    type: Literal["scale"]
    question: str
    levels: list[str] = Field(
        min_length=2,
        max_length=MAX_LEVELS,
        description="Ordered level descriptions, lowest first. Level i is returned as i.",
    )

    def to_core(self) -> TScaleQuestion:
        return TScaleQuestion(question=self.question, levels=list(self.levels))


Question = Annotated[
    Union[BinaryQuestion, ChoiceQuestion, ScaleQuestion], Field(discriminator="type")
]


class DecideRequest(BaseModel):
    input: InputValue = Field(description="The content to decide about: text, an object, or a list.")
    questions: dict[str, Question] = Field(min_length=1)
    model: Optional[str] = Field(None, description="Ignored; the server's configured model is used (unless it runs with --honor-request-model).")

    def to_core(self) -> TDecideRequest:
        questions: dict[str, TQuestion] = {k: q.to_core() for k, q in self.questions.items()}
        return TDecideRequest(input=self.input, questions=questions)


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


def answer_from_core(a: TAnswer) -> Union[BinaryAnswer, ChoiceAnswer, ScaleAnswer]:
    if isinstance(a, TBinaryAnswer):
        return BinaryAnswer(value=a.value, probability=a.probability, confidence=a.confidence)
    if isinstance(a, TChoiceAnswer):
        return ChoiceAnswer(value=a.value, probabilities=a.probabilities, confidence=a.confidence)
    return ScaleAnswer(
        value=a.value,
        level=a.level,
        probabilities=a.probabilities,
        legend=a.legend,
        confidence=a.confidence,
    )


class DecideResponse(BaseModel):
    answers: dict[str, Answer]
    meta: Meta

    @classmethod
    def from_core(cls, t: TDecideResponse) -> "DecideResponse":
        return cls(
            answers={k: answer_from_core(a) for k, a in t.answers.items()},
            meta=Meta.from_core(t.meta),
        )
