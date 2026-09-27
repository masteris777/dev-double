"""Turns a question into a single-label prompt.

Every prompt ends by asking for exactly one label (an option name, a digit, or
yes/no), so the answer sits in the first few tokens, whose probabilities we can read.

Choices are answered by option name, not by letter: small models over-pick
"B" when options are lettered, which skews the probabilities.
"""

from __future__ import annotations

import json

from .engines.base import LabelQuery
from .schemas import BinaryQuestion, ChoiceQuestion, InputValue, ScaleQuestion

SYSTEM = (
    "You are a precise decision function. Read the input, then answer the question "
    "with exactly one label from the allowed labels. Output only the label: no "
    "explanation, no punctuation, no other text."
)

def render_input(value: InputValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _frame(question: str, input_text: str, body: str) -> str:
    # Asking the question before and after the input measurably helps small
    # models: they read the input knowing what to look for.
    return f"Question: {question}\n\n<input>\n{input_text}\n</input>\n\n{body}"


def binary(input_text: str, q: BinaryQuestion) -> LabelQuery:
    lines = [f"Question: {q.question}"]
    if q.yes:
        lines.append(f'Answer "yes" if: {q.yes}')
    if q.no:
        lines.append(f'Answer "no" if: {q.no}')
    lines.append("Reply with exactly one word: yes or no.")
    return LabelQuery(
        system=SYSTEM,
        user=_frame(q.question, input_text, "\n".join(lines)),
        labels=["yes", "no"],
        input_text=input_text,
        question=q.question,
        descriptions=[f"{q.question} {q.yes or ''}", q.no or ""],
    )


def choice(input_text: str, q: ChoiceQuestion) -> LabelQuery:
    keys = list(q.options)
    option_lines = [f"- {key}: {q.options[key]}" for key in keys]
    body = (
        f"Question: {q.question}\n\nOptions:\n"
        + "\n".join(option_lines)
        + f"\n\nReply with exactly one option name: {', '.join(keys)}."
    )
    return LabelQuery(
        system=SYSTEM,
        user=_frame(q.question, input_text, body),
        labels=keys,
        input_text=input_text,
        question=q.question,
        descriptions=[f"{key} {q.options[key]}" for key in keys],
    )


def scale(input_text: str, q: ScaleQuestion) -> LabelQuery:
    digits = [str(i) for i in range(len(q.levels))]
    level_lines = [f"{d}: {text}" for d, text in zip(digits, q.levels)]
    body = (
        f"Question: {q.question}\n\nLevels (lowest first):\n"
        + "\n".join(level_lines)
        + f"\n\nReply with exactly one digit: {', '.join(digits)}."
    )
    return LabelQuery(
        system=SYSTEM,
        user=_frame(q.question, input_text, body),
        labels=digits,
        input_text=input_text,
        question=q.question,
        descriptions=list(q.levels),
    )
