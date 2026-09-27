from dev_double.core.decision.t_extract import TExtractField
from dev_double.providers.needle.decision.record_tool import record_model

FIELDS = {
    "vendor": TExtractField("string", "Company name."),
    "total": TExtractField("number"),
    "currency": TExtractField("enum", options=["EUR", "USD"]),
    "paid": TExtractField("boolean", "Already paid."),
}


def test_schema_uses_real_field_names():
    props = record_model(FIELDS).model_json_schema(by_alias=True)["properties"]
    assert list(props) == ["vendor", "total", "currency", "paid"]


def test_enum_and_boolean_fields_are_required_text_fields_optional():
    # Needle leaves optional fields without a matching text span empty, and a
    # yes/no or enum value rarely has one, so those must be required.
    required = set(record_model(FIELDS).model_json_schema(by_alias=True).get("required", []))
    assert required == {"currency", "paid"}
