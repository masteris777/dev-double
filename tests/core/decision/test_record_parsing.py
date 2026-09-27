import math

import pytest

from stunt_double.core.decision.record_parsing import (
    parse_object,
    parse_record,
    span_confidence,
    value_spans,
)
from stunt_double.core.decision.t_extract import TExtractField, TFieldValue
from stunt_double.core.decision.tracker import Tracker
from stunt_double.providers.mock.decision.clock_mock_impl import ClockMockImpl

FIELDS = {
    "vendor": TExtractField("string"),
    "total": TExtractField("number"),
    "qty": TExtractField("integer"),
}


def tracker() -> Tracker:
    return Tracker(ClockMockImpl())


# ------------------------------------------------------------------ parse_object


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Here is the JSON:\n{"a": 1}\nHope this helps.',
        '  {"a": 1}  ',
    ],
)
def test_parse_object_tolerates_fences_and_prose(text):
    assert parse_object(text) == {"a": 1}


@pytest.mark.parametrize("text", ["", "NONE", "{not json}", "[1, 2]", '{"a": 1', "} {"])
def test_parse_object_rejects_non_objects(text):
    assert parse_object(text) is None


def test_parse_object_takes_the_outermost_braces():
    assert parse_object('x {"a": {"b": "}"}} y') == {"a": {"b": "}"}}


# ------------------------------------------------------------------ value_spans


def test_value_spans_cover_raw_values_at_depth_one():
    text = 'ok {"vendor": "Acme, Inc.", "nested": {"x": [1, 2]}, "total": 12.5, "none": null}'
    spans = value_spans(text)
    assert {k: text[s:e] for k, (s, e) in spans.items()} == {
        "vendor": '"Acme, Inc."',
        "nested": '{"x": [1, 2]}',
        "total": "12.5",
        "none": "null",
    }


def test_value_spans_handle_escapes_and_whitespace():
    text = '{\n  "a\\"b": "say \\"hi\\"" ,\n  "c": 1\n}'
    spans = value_spans(text)
    assert text[slice(*spans['a"b'])] == '"say \\"hi\\""'
    assert text[slice(*spans["c"])] == "1"
    assert value_spans("no object") == {}


# ------------------------------------------------------------------ span_confidence


def test_span_confidence_uses_overlapping_tokens():
    tokens = [("ab", math.log(0.5)), ("cd", math.log(0.8)), ("ef", math.log(0.2))]
    assert span_confidence(tokens, 2, 4) == pytest.approx(0.8)
    assert span_confidence(tokens, 1, 3) == pytest.approx(math.sqrt(0.5 * 0.8))
    assert span_confidence(tokens, 6, 8) is None
    assert span_confidence([], 0, 1) is None


# ------------------------------------------------------------------ parse_record


def lp(p: float) -> float:
    return math.log(p)


def test_record_with_per_field_confidence_from_tokens():
    tokens = [
        ('{"vendor": ', lp(0.99)),
        ('"Acme', lp(0.9)),
        (' Corp"', lp(0.9)),
        (', "total": ', lp(0.99)),
        ('"1.200,50"', lp(0.6)),
        (', "qty": ', lp(0.99)),
        ("null", lp(0.7)),
        ("}", lp(0.99)),
    ]
    text = "".join(t for t, _ in tokens)
    t = tracker()
    got = parse_record(text, tokens, FIELDS, fallback_confidence=0.5, tracker=t)
    assert got == {
        "vendor": TFieldValue("Acme Corp", 0.9),
        "total": TFieldValue(1200.5, 0.6),
        "qty": TFieldValue(None, 0.7),
    }
    assert t.warnings == []


def test_record_without_tokens_uses_the_fallback_confidence():
    text = '```json\n{"vendor": "Acme", "total": 1200.5, "qty": 3.0}\n```'
    got = parse_record(text, [], FIELDS, fallback_confidence=0.8, tracker=tracker())
    assert got == {
        "vendor": TFieldValue("Acme", 0.8),
        "total": TFieldValue(1200.5, 0.8),
        "qty": TFieldValue(3, 0.8),
    }
    assert isinstance(got["qty"].value, int)


def test_missing_keys_are_none_and_unmatched_spans_fall_back():
    # Tokens that don't spell the JSON (e.g. the server re-encoded it): spans aren't found.
    got = parse_record('{"vendor": "Acme"}', [("x", lp(0.9))], FIELDS, fallback_confidence=0.9, tracker=tracker())
    assert got["vendor"] == TFieldValue("Acme", 0.9)
    assert got["total"] == TFieldValue(None, 0.9) and got["qty"] == TFieldValue(None, 0.9)


def test_european_and_currency_number_strings():
    text = '{"vendor": "A", "total": "EUR 1.234.567,89", "qty": "1,000"}'
    got = parse_record(text, [], FIELDS, fallback_confidence=1.0, tracker=tracker())
    assert got["total"].value == pytest.approx(1234567.89) and got["qty"].value == 1000


def test_unparseable_values_warn_and_are_empty():
    t = tracker()
    text = '{"vendor": ["a", "b"], "total": "lots", "qty": 2.5}'
    got = parse_record(text, [], FIELDS, fallback_confidence=1.0, tracker=t)
    assert got["total"] == TFieldValue(None, 0.0) and got["qty"] == TFieldValue(None, 0.0)
    assert got["vendor"].value == '["a", "b"]'
    assert [w.split(":")[0] for w in t.warnings] == ["total", "qty"]


def test_string_none_markers_are_none():
    got = parse_record('{"vendor": "N/A"}', [], {"vendor": TExtractField("string")}, 0.7, tracker())
    assert got["vendor"] == TFieldValue(None, 0.7)


def test_invalid_json_empties_every_field_with_a_warning():
    t = tracker()
    got = parse_record("I could not find it.", [], FIELDS, fallback_confidence=1.0, tracker=t)
    assert got == {name: TFieldValue(None, 0.0) for name in FIELDS}
    assert len(t.warnings) == 1 and "JSON" in t.warnings[0]
