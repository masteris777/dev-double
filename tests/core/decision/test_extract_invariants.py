import pytest

from dev_double.core.decision.t_extract import (
    MAX_FIELDS,
    TExtractField,
    TExtractRequest,
    TExtractResponse,
    TFieldValue,
)
from dev_double.core.decision.t_meta import TMeta
from dev_double.core.decision.t_question import MAX_OPTIONS
from dev_double.core.decision.t_usage import TUsage


def test_field_types():
    for kind in ("string", "number", "integer", "boolean"):
        assert TExtractField(kind).options is None
    with pytest.raises(ValueError, match="type"):
        TExtractField("date")


def test_enum_needs_options():
    with pytest.raises(ValueError, match="options"):
        TExtractField("enum")
    with pytest.raises(ValueError):
        TExtractField("enum", options=["only"])
    with pytest.raises(ValueError):
        TExtractField("enum", options=[f"o{i}" for i in range(MAX_OPTIONS + 1)])


def test_enum_options_list_becomes_a_dict_of_names():
    f = TExtractField("enum", options=["EUR", "USD"])
    assert f.options == {"EUR": "EUR", "USD": "USD"}
    f = TExtractField("enum", "The currency.", {"EUR": "Euro", "USD": "US dollar"})
    assert f.options == {"EUR": "Euro", "USD": "US dollar"} and f.description == "The currency."


def test_enum_options_must_be_distinct():
    with pytest.raises(ValueError, match="uplicate"):
        TExtractField("enum", options=["EUR", "EUR"])
    with pytest.raises(ValueError, match="too similar"):
        TExtractField("enum", options=["in_stock", "In-Stock"])


def test_only_enum_takes_options():
    with pytest.raises(ValueError, match="options"):
        TExtractField("string", options=["a", "b"])


def test_request_needs_1_to_30_named_fields():
    with pytest.raises(ValueError):
        TExtractRequest("x", {})
    with pytest.raises(ValueError):
        TExtractRequest("x", {f"f{i}": TExtractField("string") for i in range(MAX_FIELDS + 1)})
    with pytest.raises(ValueError, match="name"):
        TExtractRequest("x", {" ": TExtractField("string")})
    TExtractRequest("x", {f"f{i}": TExtractField("string") for i in range(MAX_FIELDS)})


def test_field_value_invariants():
    TFieldValue(None, 0.0)
    with pytest.raises(ValueError):
        TFieldValue("x", 1.5)
    with pytest.raises(ValueError):
        TFieldValue(True, 0.5, probability=-0.1)


def test_response_values_follow_fields():
    meta = TMeta("e", "m", 0, TUsage())
    r = TExtractResponse({"a": TFieldValue("x", 0.9), "b": TFieldValue(None, 0.7)}, meta)
    assert r.values == {"a": "x", "b": None}
