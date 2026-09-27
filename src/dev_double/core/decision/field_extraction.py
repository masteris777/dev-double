"""Reads every field of an extraction request with whatever the decider can do.

- A native ``IExtractor`` (e.g. one tool call per record) takes the whole request.
- Otherwise enum and boolean fields are asked as choice and binary questions
  (real probabilities), and all string, number, and integer fields go to one
  ``IRecordReader.read_record`` call when the decider has it (a whole-record
  prompt beat one prompt per field in our evals), else they are left empty
  with a warning.

The questions and the record read run concurrently; results keep the
request's field order.
"""

from __future__ import annotations

import asyncio

from .extract_questions import boolean_question, cannot_generate, enum_question, value_from_answer
from .i_decider import IDecider
from .i_extractor import IExtractor
from .i_record_reader import IRecordReader
from .t_extract import EMPTY, TEXT_TYPES, TExtractField, TExtractRequest, TFieldValue
from .t_input import TInputValue
from .tracker import Tracker


async def _ask(decider: IDecider, value: TInputValue, name: str, field: TExtractField, t: Tracker) -> TFieldValue:
    question = enum_question(name, field) if field.type == "enum" else boolean_question(name, field)
    return value_from_answer(await decider.ask(value, question, t, label=name))


async def _read_text(
    decider: IDecider, value: TInputValue, fields: dict[str, TExtractField], t: Tracker
) -> dict[str, TFieldValue]:
    if not fields:
        return {}
    if isinstance(decider, IRecordReader):
        return await decider.read_record(value, fields, t)
    return {name: cannot_generate(name, t) for name in fields}


async def read_fields(decider: IDecider, req: TExtractRequest, t: Tracker) -> dict[str, TFieldValue]:
    if isinstance(decider, IExtractor):
        got = await decider.extract(req, t)
        return {name: got.get(name, EMPTY) for name in req.fields}
    text = {name: f for name, f in req.fields.items() if f.type in TEXT_TYPES}
    scored = {name: f for name, f in req.fields.items() if f.type not in TEXT_TYPES}
    record, *answers = await asyncio.gather(
        _read_text(decider, req.input, text, t),
        *(_ask(decider, req.input, name, f, t) for name, f in scored.items()),
    )
    got = {**record, **dict(zip(scored, answers))}
    return {name: got.get(name, EMPTY) for name in req.fields}
