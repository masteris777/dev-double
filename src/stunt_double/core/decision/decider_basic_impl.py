"""The default decider: renders a single-label prompt, asks an ``IEngine`` for
a distribution over the labels, and shapes the typed answer."""

from __future__ import annotations

import asyncio

from . import prompts
from .answer_shaping import shape_answer
from .i_decider import IDecider
from .i_engine import IEngine
from .t_answer import TAnswer
from .t_input import TInputValue
from .t_question import TQuestion
from .tracker import Tracker


class DeciderBasicImpl(IDecider):
    def __init__(self, engine: IEngine, max_concurrency: int = 4) -> None:
        self.engine = engine
        self._limit = asyncio.Semaphore(max_concurrency)

    @property
    def name(self) -> str:
        return self.engine.name

    @property
    def model(self) -> str:
        return self.engine.model

    async def ask(
        self, value: TInputValue, question: TQuestion, tracker: Tracker, label: str = ""
    ) -> TAnswer:
        query = prompts.query_for(prompts.render_input(value), question)
        async with self._limit:
            result = await self.engine.distribution(query)
        tracker.usage.add(result.usage)
        for warning in result.warnings:
            tracker.warn(warning, label)
        return shape_answer(question, result.probabilities)

    async def aclose(self) -> None:
        await self.engine.aclose()
