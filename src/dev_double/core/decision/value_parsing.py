"""Pure helpers that turn a model's short free-text answer into a typed value.

Models add noise around the value: quotes, a trailing period, currency
symbols, thousands separators. We strip it rather than reject the answer.
"""

from __future__ import annotations

import json
from typing import Optional

from .t_extract import TEXT_TYPES, TValue

NONE_MARKERS = frozenset({"none", "n/a", "null", ""})
_QUOTES = "\"'`"
_DIGITS = "0123456789"
_MINUS = "-−"
_CURRENCY = "$€£¥₹¢"


def clean_text(text: str) -> str:
    """Strip whitespace, surrounding quotes, and one trailing period."""
    s = text.strip().strip(_QUOTES).strip()
    if s.endswith("."):
        s = s[:-1].rstrip()
    return s.strip(_QUOTES).strip()


def _is_negative(before: str) -> bool:
    """A minus sign right before the number, allowing spaces or a currency symbol between ("-$5")."""
    for ch in reversed(before):
        if ch in _MINUS:
            return True
        if not (ch.isspace() or ch in _CURRENCY):
            return False
    return False


def _normalize_separators(run: str) -> str:
    """Digits with "," and "." as thousands or decimal separators -> a float literal.

    When both appear, the last one is the decimal separator ("1,200.50", "1.200,50").
    A lone comma is a decimal comma unless exactly three digits follow it ("12,50" vs "1,200").
    Several dots are thousands separators ("1.200.000").
    """
    comma, dot = run.rfind(","), run.rfind(".")
    if comma >= 0 and dot >= 0:
        if comma > dot:
            return run.replace(".", "").replace(",", ".")
        return run.replace(",", "")
    if comma >= 0:
        if run.count(",") == 1 and len(run) - comma - 1 != 3:
            return run.replace(",", ".")
        return run.replace(",", "")
    if run.count(".") > 1:
        return run.replace(".", "")
    return run


def parse_number(text: str) -> Optional[float]:
    """The first number in the text, or None if it has none."""
    start = next((i for i, ch in enumerate(text) if ch in _DIGITS), None)
    if start is None:
        return None
    end = start
    while end < len(text) and (text[end] in _DIGITS or text[end] in ".,"):
        end += 1
    run = text[start:end].rstrip(".,")
    try:
        value = float(_normalize_separators(run))
    except ValueError:
        return None
    return -value if _is_negative(text[:start]) else value


def coerce_json_value(raw: object, kind: str) -> tuple[TValue, Optional[str]]:
    """(value, warning) for a JSON value of a string, number, or integer field.

    Numbers pass through directly; anything else is read as text with ``parse_field_text``.
    """
    if kind in ("number", "integer") and isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if kind == "number":
            return float(raw), None
        return (int(raw), None) if float(raw).is_integer() else (None, f"{raw!r} is not a whole number; left empty")
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    return parse_field_text(text, kind)


def parse_field_text(text: str, kind: str) -> tuple[TValue, Optional[str]]:
    """(value, warning) for a string, number, or integer field.

    "NONE" and similar markers mean the input doesn't contain the value: (None, None).
    An answer that can't be read as the type gives (None, a warning).
    """
    if kind not in TEXT_TYPES:
        raise ValueError(f"Only {', '.join(TEXT_TYPES)} fields are parsed from text, not {kind!r}.")
    s = clean_text(text)
    if s.lower() in NONE_MARKERS:
        return None, None
    if kind == "string":
        return s, None
    number = parse_number(s)
    if number is None:
        return None, f"could not read a number from {s[:40]!r}; left empty"
    if kind == "integer":
        if not number.is_integer():
            return None, f"{s[:40]!r} is not a whole number; left empty"
        return int(number), None
    return number, None
