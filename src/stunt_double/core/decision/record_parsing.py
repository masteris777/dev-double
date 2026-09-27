"""Pure helpers that turn a model's JSON record into typed field values, with a
per-field confidence read from the tokens that spell each value.

The model may wrap the object in ``` fences or prose; we take the outermost
{...}. Each value's span in the output is located by a small depth-aware
scan, and its confidence is exp(mean logprob) of the tokens overlapping it.
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional, Sequence

from .confidence import DIGITS
from .t_extract import EMPTY, TExtractField, TFieldValue
from .tracker import Tracker
from .value_parsing import coerce_json_value


def parse_object(text: str) -> Optional[dict[str, Any]]:
    """The outermost {...} in the text as a dict, or None if it isn't a JSON object."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _string_end(text: str, i: int) -> int:
    """Index just past the JSON string starting at text[i] == '"', or -1."""
    j = i + 1
    while j < len(text):
        if text[j] == "\\":
            j += 2
            continue
        if text[j] == '"':
            return j + 1
        j += 1
    return -1


def _value_end(text: str, j: int) -> int:
    """Index of the ',' or '}' that ends the value starting at j (depth 0, outside strings)."""
    depth = 0
    while j < len(text):
        ch = text[j]
        if ch == '"':
            end = _string_end(text, j)
            if end < 0:
                return len(text)
            j = end
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            if depth == 0:
                return j
            depth -= 1
        elif ch == "," and depth == 0:
            return j
        j += 1
    return j


def value_spans(text: str) -> dict[str, tuple[int, int]]:
    """Key -> (start, end) of its raw value (e.g. '"Acme"', '12.5', 'null') in the
    first top-level object of the text. Stops quietly at malformed JSON."""
    i = text.find("{")
    if i < 0:
        return {}
    spans: dict[str, tuple[int, int]] = {}
    i += 1
    while i < len(text):
        if text[i].isspace() or text[i] == ",":
            i += 1
            continue
        if text[i] != '"':
            break  # '}' or malformed
        key_end = _string_end(text, i)
        if key_end < 0:
            break
        try:
            key = json.loads(text[i:key_end])
        except ValueError:
            break
        j = key_end
        while j < len(text) and text[j].isspace():
            j += 1
        if j >= len(text) or text[j] != ":":
            break
        j += 1
        while j < len(text) and text[j].isspace():
            j += 1
        end = _value_end(text, j)
        stop = end
        while stop > j and text[stop - 1].isspace():
            stop -= 1
        spans[key] = (j, stop)
        i = end
    return spans


def span_confidence(tokens: Sequence[tuple[str, float]], start: int, end: int) -> Optional[float]:
    """exp(mean logprob) of the tokens overlapping text[start:end], or None if none do."""
    logprobs, offset = [], 0
    for token, logprob in tokens:
        if offset < end and offset + len(token) > start:
            logprobs.append(logprob)
        offset += len(token)
    if not logprobs:
        return None
    return min(1.0, math.exp(sum(logprobs) / len(logprobs)))


def parse_record(
    text: str,
    tokens: Sequence[tuple[str, float]],
    fields: dict[str, TExtractField],
    fallback_confidence: float,
    tracker: Tracker,
) -> dict[str, TFieldValue]:
    """Field name -> value for the text fields in a generated JSON record.

    Missing keys and nulls give None. A value's confidence comes from its own
    tokens; ``fallback_confidence`` (the whole generation's) is used when there
    are no tokens or the value's span can't be found in them.
    """
    obj = parse_object(text)
    if obj is None:
        tracker.warn(f"the model did not return a JSON object (got {text[:40]!r}); text fields left empty")
        return {name: EMPTY for name in fields}
    spans = value_spans("".join(token for token, _ in tokens)) if tokens else {}
    out: dict[str, TFieldValue] = {}
    for name, field in fields.items():
        conf = None
        if name in spans:
            conf = span_confidence(tokens, *spans[name])
        conf = round(fallback_confidence if conf is None else conf, DIGITS)
        raw = obj.get(name)
        if raw is None:
            out[name] = TFieldValue(None, conf)
            continue
        value, problem = coerce_json_value(raw, field.type)
        if problem is not None:
            tracker.warn(problem, name)
            out[name] = EMPTY
            continue
        out[name] = TFieldValue(value, conf)
    return out
