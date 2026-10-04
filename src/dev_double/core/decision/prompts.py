"""Turns a question into a single-label prompt.

Every prompt ends by asking for exactly one label (an option name, a digit, or
yes/no), so the answer sits in the first few tokens, whose probabilities we can read.

Choices are answered by option name, not by letter: small models over-pick
"B" when options are lettered, which skews the probabilities.
"""

from __future__ import annotations

import json

from .t_extract import TExtractField
from .t_generate import TGenerateQuery
from .t_input import TInputValue
from .t_label_query import TLabelQuery
from .t_question import TBinaryQuestion, TChoiceQuestion, TQuestion, TScaleQuestion

SYSTEM = (
    "You are a precise decision function. Read the input, then answer the question "
    "with exactly one label from the allowed labels. Output only the label: no "
    "explanation, no punctuation, no other text."
)

# Extraction: all free-text fields of a request in one JSON object. Wording
# chosen by A/B test against a per-field prompt (which small models answered
# with NONE for values plainly present, or echoed the field description).
RECORD_SYSTEM = (
    "You extract data from text into JSON. Output one JSON object with exactly the requested "
    "keys. Copy values from the text, converting them to the requested format. Use null only "
    "when the text gives no information for a key. Output only the JSON."
)
RECORD_MAX_TOKENS = 512

def render_input(value: TInputValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _frame(question: str, input_text: str, body: str) -> str:
    # Asking the question before and after the input measurably helps small
    # models: they read the input knowing what to look for.
    return f"Question: {question}\n\n<input>\n{input_text}\n</input>\n\n{body}"


def binary(input_text: str, q: TBinaryQuestion) -> TLabelQuery:
    lines = [f"Question: {q.question}"]
    if q.yes:
        lines.append(f'Answer "yes" if: {q.yes}')
    if q.no:
        lines.append(f'Answer "no" if: {q.no}')
    lines.append("Reply with exactly one word: yes or no.")
    return TLabelQuery(
        system=SYSTEM,
        user=_frame(q.question, input_text, "\n".join(lines)),
        labels=["yes", "no"],
        input_text=input_text,
        question=q.question,
        descriptions=[f"{q.question} {q.yes or ''}", q.no or ""],
    )


def choice(input_text: str, q: TChoiceQuestion) -> TLabelQuery:
    keys = list(q.options)
    option_lines = [f"- {key}: {q.options[key]}" for key in keys]
    body = (
        f"Question: {q.question}\n\nOptions:\n"
        + "\n".join(option_lines)
        + f"\n\nReply with exactly one option name: {', '.join(keys)}."
    )
    return TLabelQuery(
        system=SYSTEM,
        user=_frame(q.question, input_text, body),
        labels=keys,
        input_text=input_text,
        question=q.question,
        descriptions=[f"{key} {q.options[key]}" for key in keys],
    )


def scale(input_text: str, q: TScaleQuestion) -> TLabelQuery:
    digits = [str(i) for i in range(len(q.levels))]
    word = "digit" if len(digits) <= 10 else "number"
    level_lines = [f"{d}: {text}" for d, text in zip(digits, q.levels)]
    body = (
        f"Question: {q.question}\n\nLevels (lowest first):\n"
        + "\n".join(level_lines)
        + f"\n\nReply with exactly one {word}: {', '.join(digits)}."
    )
    return TLabelQuery(
        system=SYSTEM,
        user=_frame(q.question, input_text, body),
        labels=digits,
        input_text=input_text,
        question=q.question,
        descriptions=list(q.levels),
    )


def query_for(input_text: str, q: TQuestion) -> TLabelQuery:
    """Dispatch on the question type."""
    if isinstance(q, TBinaryQuestion):
        return binary(input_text, q)
    if isinstance(q, TChoiceQuestion):
        return choice(input_text, q)
    return scale(input_text, q)


def record_max_tokens(n_fields: int) -> int:
    return min(RECORD_MAX_TOKENS, 48 + 40 * n_fields)


def record_key_line(name: str, field: TExtractField) -> str:
    return f'- "{name}" ({field.type}): {field.description or name.replace("_", " ")}'


def extract_record(input_text: str, fields: dict[str, TExtractField]) -> TGenerateQuery:
    """Free-text fields (string, number, integer) as one JSON object, keys listed
    before and after the text."""
    keys = "\n".join(record_key_line(name, f) for name, f in fields.items())
    user = f"Keys:\n{keys}\n\n<text>\n{input_text}\n</text>\n\nFill in the keys from the text above:\n{keys}\n\nJSON:"
    return TGenerateQuery(
        system=RECORD_SYSTEM,
        user=user,
        max_tokens=record_max_tokens(len(fields)),
        json=True,
        input_text=input_text,
        fields=tuple(fields),
    )
