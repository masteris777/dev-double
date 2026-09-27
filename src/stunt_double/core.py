"""Runs questions against an engine and shapes the answers."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from . import prompts
from .engines.base import Engine
from .schemas import (
    Answer,
    BinaryAnswer,
    BinaryQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    InputValue,
    Meta,
    Question,
    ScaleAnswer,
    ScaleQuestion,
    Usage,
)

DIGITS = 4


def confidence(probabilities: list[float]) -> float:
    """How concentrated a distribution is: 1.0 when all mass is on one label,
    0.0 when it is spread evenly. Normalized max probability: (n*max - 1) / (n - 1).
    """
    n = len(probabilities)
    if n < 2:
        return 1.0
    return max(0.0, (n * max(probabilities) - 1) / (n - 1))


def _round(probs: dict[str, float]) -> dict[str, float]:
    return {k: round(v, DIGITS) for k, v in probs.items()}


@dataclass
class Tracker:
    """Collects usage and warnings across the engine calls of one request."""

    usage: Usage = field(default_factory=Usage)
    warnings: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)

    def meta(self, engine: Engine) -> Meta:
        return Meta(
            engine=engine.name,
            model=engine.model,
            latency_ms=round((time.perf_counter() - self.started) * 1000),
            usage=self.usage,
            warnings=list(dict.fromkeys(self.warnings)),
        )


class Decider:
    def __init__(self, engine: Engine, max_concurrency: int = 4) -> None:
        self.engine = engine
        self._limit = asyncio.Semaphore(max_concurrency)

    async def ask(
        self, value: InputValue, question: Question, tracker: Tracker, label: str = ""
    ) -> Answer:
        input_text = prompts.render_input(value)
        if isinstance(question, BinaryQuestion):
            query = prompts.binary(input_text, question)
        elif isinstance(question, ChoiceQuestion):
            query = prompts.choice(input_text, question)
        else:
            query = prompts.scale(input_text, question)

        async with self._limit:
            result = await self.engine.distribution(query)
        tracker.usage.add(result.usage)
        prefix = f"{label}: " if label else ""
        tracker.warnings.extend(prefix + w for w in result.warnings)
        probs = result.probabilities

        if isinstance(question, BinaryQuestion):
            p_yes = probs["yes"]
            return BinaryAnswer(
                value=p_yes >= 0.5,
                probability=round(p_yes, DIGITS),
                confidence=round(confidence([p_yes, 1 - p_yes]), DIGITS),
            )

        if isinstance(question, ChoiceQuestion):
            return ChoiceAnswer(
                value=max(probs, key=probs.__getitem__),
                probabilities=_round(probs),
                confidence=round(confidence(list(probs.values())), DIGITS),
            )

        assert isinstance(question, ScaleQuestion)
        expected = sum(int(level) * p for level, p in probs.items())
        return ScaleAnswer(
            value=round(expected, DIGITS),
            level=int(max(probs, key=probs.__getitem__)),
            probabilities=_round(probs),
            legend={str(i): text for i, text in enumerate(question.levels)},
            confidence=round(confidence(list(probs.values())), DIGITS),
        )

    async def ask_many(
        self, value: InputValue, questions: dict[str, Question], tracker: Tracker
    ) -> dict[str, Answer]:
        answers = await asyncio.gather(
            *(self.ask(value, q, tracker, label=qid) for qid, q in questions.items())
        )
        return dict(zip(questions, answers))
