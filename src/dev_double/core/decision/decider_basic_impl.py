"""The default decider: renders a single-label prompt, asks an ``IEngine`` for
a distribution over the labels, and shapes the typed answer.

It also reads the free-text fields of an extraction request, as one JSON
record, when its engine is an ``IGenerator``; with a label-only engine those
fields are left empty with a warning."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from . import prompts
from .answer_shaping import shape_answer
from .extract_questions import cannot_generate
from .i_decider import IDecider
from .i_engine import IEngine
from .i_generator import IGenerator
from .i_record_reader import IRecordReader
from .record_parsing import parse_record
from .t_answer import TAnswer
from .t_extract import TExtractField, TFieldValue
from .t_input import TInputValue
from .t_question import TQuestion
from .tracker import Tracker


class DeciderBasicImpl(IDecider, IRecordReader):
    def __init__(self, engine: IEngine, max_concurrency: int = 4) -> None:
        self.engine = engine
        self._limit = asyncio.Semaphore(max_concurrency)

    @property
    def name(self) -> str:
        return self.engine.name

    @property
    def model(self) -> str:
        return self.engine.model

    @property
    def supports_images(self) -> bool:
        return self.engine.supports_images

    def model_for(self, images: bool) -> str:
        return self.engine.model_for(images)

    async def ask(
        self,
        value: TInputValue,
        question: TQuestion,
        tracker: Tracker,
        label: str = "",
        images: tuple[str, ...] = (),
    ) -> TAnswer:
        query = replace(prompts.query_for(prompts.render_input(value), question), images=images)
        async with self._limit:
            result = await self.engine.distribution(query)
        tracker.usage.add(result.usage)
        for warning in result.warnings:
            tracker.warn(warning, label)
        return shape_answer(question, result.probabilities)

    async def read_record(
        self, input_value: TInputValue, fields: dict[str, TExtractField], tracker: Tracker
    ) -> dict[str, TFieldValue]:
        if not isinstance(self.engine, IGenerator):
            return {name: cannot_generate(name, tracker) for name in fields}
        query = prompts.extract_record(prompts.render_input(input_value), fields)
        async with self._limit:
            result = await self.engine.generate(query)
        tracker.usage.add(result.usage)
        for warning in result.warnings:
            tracker.warn(warning)
        return parse_record(result.text, result.tokens, fields, result.confidence, tracker)

    async def aclose(self) -> None:
        await self.engine.aclose()
