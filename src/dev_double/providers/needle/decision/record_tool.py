"""Extraction as one Needle tool call: every field becomes an argument
of a ``record`` tool, and the call's arguments are mapped back to typed values.

pydantic is imported lazily, like needle, so this module imports without either.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from dev_double.core.decision.confidence import DIGITS
from dev_double.core.decision.t_extract import EMPTY, TExtractField, TFieldValue
from dev_double.core.decision.tracker import Tracker
from dev_double.core.decision.value_parsing import coerce_json_value

_PY_TYPES: dict[str, type] = {"string": str, "number": float, "integer": int, "boolean": bool}

RECORD_SYSTEM = (
    "Extract the described fields from the input. Always call record once. "
    "Leave out a text or number field the input does not contain."
)


def fields_key(fields: dict[str, TExtractField]) -> tuple:
    """A hashable key for a field set, to reuse one agent per distinct set."""
    return tuple((n, f.type, f.description, tuple((f.options or {}).items())) for n, f in fields.items())  # type: ignore[union-attr]


def record_model(fields: dict[str, TExtractField]) -> Any:
    """The tool schema: field names are aliases, so any name (even "due date") works."""
    from pydantic import Field, create_model

    specs: dict[str, Any] = {}
    for i, (name, f) in enumerate(fields.items()):
        description = f.description or name
        if f.type == "enum":
            assert isinstance(f.options, dict)
            kind: Any = Literal[tuple(f.options)]  # type: ignore[valid-type]
            description += " One of: " + "; ".join(f"{k}: {v}" for k, v in f.options.items())
        else:
            kind = _PY_TYPES[f.type]
        if f.type in ("enum", "boolean"):
            # Always answerable from the text, so required: optional fields with no
            # matching span come back empty from Needle.
            specs[f"f{i}"] = (kind, Field(..., alias=name, description=description))
        else:
            specs[f"f{i}"] = (Optional[kind], Field(None, alias=name, description=description))
    return create_model("record", __doc__="Record the fields found in the input.", **specs)


def _coerce(f: TExtractField, raw: Any) -> tuple[Any, Optional[str]]:
    if f.type == "boolean":
        return (raw, None) if isinstance(raw, bool) else (None, f"not a boolean: {raw!r}")
    if f.type == "enum":
        return (raw, None) if raw in (f.options or {}) else (None, f"not one of the options: {raw!r}")
    return coerce_json_value(raw, f.type)


def value_from_argument(name: str, f: TExtractField, raw: Any, conf: float, tracker: Tracker) -> TFieldValue:
    """One argument of the call as a field value; the call's confidence applies to every field."""
    if raw is None:
        return TFieldValue(None, conf)
    value, problem = _coerce(f, raw)
    if problem is not None:
        tracker.warn(problem, name)
        return EMPTY
    if value is None:
        return TFieldValue(None, conf)
    if f.type == "enum":
        rest = round((1 - conf) / (len(f.options or {}) - 1), DIGITS)
        probs = {k: (conf if k == value else rest) for k in f.options or {}}
        return TFieldValue(value, conf, probabilities=probs)
    if f.type == "boolean":
        return TFieldValue(value, conf, probability=round(conf if value else 1 - conf, DIGITS))
    return TFieldValue(value, conf)
