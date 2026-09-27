import pytest
from fastapi.testclient import TestClient

from stunt_double.apps.config import Settings
from stunt_double.apps.server.app import create_app

INVOICE = "INVOICE\nVendor: Acme Corp\nTotal: 1,200.50\nCurrency: euro, paid in euros\nStatus: already paid"

FIELDS = {
    "vendor": {"type": "string", "description": "Company that issued the invoice."},
    "total": {"type": "number", "description": "Amount due, without currency symbol."},
    "due_date": {"type": "string", "description": "Due date as YYYY-MM-DD."},
    "currency": {"type": "enum", "options": {"EUR": "euro euros", "USD": "US dollars", "GBP": "pounds"}},
    "paid": {"type": "boolean", "description": "Whether the invoice is already paid."},
}


@pytest.fixture
def client():
    with TestClient(create_app(Settings(engine="mock"))) as c:
        yield c


def test_extract_shape(client):
    r = client.post("/v1/extract", json={"input": INVOICE, "fields": FIELDS})
    assert r.status_code == 200, r.text
    body = r.json()
    values, fields = body["values"], body["fields"]
    assert list(values) == list(FIELDS) and list(fields) == list(FIELDS)
    assert values["vendor"] == "Acme Corp" and values["total"] == 1200.5
    assert values["due_date"] is None and values["currency"] == "EUR" and isinstance(values["paid"], bool)
    # None values stay as null; None probabilities are omitted.
    assert fields["due_date"] == {"value": None, "confidence": 1.0}
    assert set(fields["vendor"]) == {"value", "confidence"}
    assert set(fields["currency"]) == {"value", "confidence", "probabilities"}
    assert set(fields["paid"]) == {"value", "confidence", "probability"}
    assert body["meta"]["engine"] == "mock"


def test_enum_options_may_be_a_list_and_input_an_object(client):
    body = {"input": {"currency": "USD"}, "fields": {"currency": {"type": "enum", "options": ["EUR", "USD"]}}}
    r = client.post("/v1/extract", json=body)
    assert r.status_code == 200, r.text
    assert set(r.json()["fields"]["currency"]["probabilities"]) == {"EUR", "USD"}


def test_extract_422s(client):
    def post(fields):  # type: ignore[no-untyped-def]
        return client.post("/v1/extract", json={"input": "x", "fields": fields})

    r = post({"currency": {"type": "enum"}})
    assert r.status_code == 422 and "options" in r.text
    assert r.json()["detail"][0]["loc"][:3] == ["body", "fields", "currency"]
    r = post({"currency": {"type": "enum", "options": ["EUR", "EUR"]}})
    assert r.status_code == 422 and "uplicate" in r.text
    assert post({}).status_code == 422
    assert post({"x": {"type": "date"}}).status_code == 422
    assert post({"x": {"type": "string", "options": ["a", "b"]}}).status_code == 422
    assert post({"": {"type": "string"}}).status_code == 422
    assert post({f"f{i}": {"type": "string"} for i in range(31)}).status_code == 422


def test_openapi_lists_extract(client):
    assert "/v1/extract" in client.get("/openapi.json").json()["paths"]
