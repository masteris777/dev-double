"""How extraction fields map onto the three question types.

Enum fields become choice questions and boolean fields binary questions, so
they work with every decider and carry real probabilities.
"""

from __future__ import annotations

from .t_answer import TAnswer, TBinaryAnswer, TChoiceAnswer
from .t_extract import EMPTY, TExtractField, TFieldValue
from .t_question import TBinaryQuestion, TChoiceQuestion
from .tracker import Tracker


def field_question_text(name: str, field: TExtractField) -> str:
    question = f'What is the value of the field "{name}" in the input?'
    return f"{question} {field.description}" if field.description else question


def enum_question(name: str, field: TExtractField) -> TChoiceQuestion:
    assert isinstance(field.options, dict)  # TExtractField normalizes enum options to a dict
    return TChoiceQuestion(question=field_question_text(name, field), options=dict(field.options))


def boolean_question(name: str, field: TExtractField) -> TBinaryQuestion:
    return TBinaryQuestion(question=f"Is this true for the input? {field.description or name}")


def value_from_answer(answer: TAnswer) -> TFieldValue:
    if isinstance(answer, TBinaryAnswer):
        return TFieldValue(answer.value, answer.confidence, probability=answer.probability)
    if isinstance(answer, TChoiceAnswer):
        return TFieldValue(answer.value, answer.confidence, probabilities=answer.probabilities)
    raise TypeError(f"No field value for a {type(answer).__name__}.")


def cannot_generate(name: str, tracker: Tracker) -> TFieldValue:
    """The field needs free text and the engine can't generate any."""
    tracker.warn(f"engine can't generate text: field {name} left empty")
    return EMPTY
