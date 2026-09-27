import pytest

from stunt_double.core.decision.value_parsing import clean_text, parse_field_text, parse_number


@pytest.mark.parametrize(
    "raw, cleaned",
    [
        ("  Acme Corp  ", "Acme Corp"),
        ('"Acme Corp"', "Acme Corp"),
        ("'2026-10-01'", "2026-10-01"),
        ("Acme Corp.", "Acme Corp"),
        ('"Acme Corp."\n', "Acme Corp"),
        ("`x`", "x"),
    ],
)
def test_clean_text(raw, cleaned):
    assert clean_text(raw) == cleaned


@pytest.mark.parametrize("raw", ["NONE", "none", " None. ", "N/A", "n/a", "null", "", "   ", '""'])
def test_none_markers_become_none_without_warning(raw):
    for kind in ("string", "number", "integer"):
        assert parse_field_text(raw, kind) == (None, None)


def test_string_is_returned_cleaned():
    assert parse_field_text(' "Acme Corp" ', "string") == ("Acme Corp", None)


@pytest.mark.parametrize(
    "raw, value",
    [
        ("1200", 1200.0),
        ("1200.50", 1200.5),
        ("1,200.50", 1200.5),
        ("1,234,567", 1234567.0),
        ("1.200,50", 1200.5),
        ("1.234.567,89", 1234567.89),
        ("1.200.000", 1200000.0),
        ("12,50", 12.5),
        ("$1,200.50", 1200.5),
        ("€ 99", 99.0),
        ("99 EUR", 99.0),
        ("-42.5", -42.5),
        ("-$5", -5.0),
        ("−3", -3.0),
        ("Total: 7.", 7.0),
        ("0.5", 0.5),
    ],
)
def test_parse_number(raw, value):
    assert parse_number(raw) == pytest.approx(value)


@pytest.mark.parametrize("raw", ["abc", "-", "..", "$"])
def test_parse_number_without_digits_is_none(raw):
    assert parse_number(raw) is None


def test_number_field():
    assert parse_field_text("USD 1,200.50", "number") == (1200.5, None)
    value, warning = parse_field_text("twelve", "number")
    assert value is None and "number" in warning and "twelve" in warning


def test_integer_field():
    assert parse_field_text("3", "integer") == (3, None)
    assert parse_field_text("1,000", "integer") == (1000, None)
    assert isinstance(parse_field_text("3.0", "integer")[0], int)
    value, warning = parse_field_text("2.5", "integer")
    assert value is None and "whole number" in warning
    value, warning = parse_field_text("many", "integer")
    assert value is None and warning


def test_other_types_are_rejected():
    with pytest.raises(ValueError):
        parse_field_text("x", "enum")
