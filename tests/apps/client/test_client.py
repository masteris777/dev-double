from fastapi.testclient import TestClient

from dev_double.apps.client import DevDouble as AppsDevDouble
from dev_double.apps.config import Settings
from dev_double.apps.server.app import create_app
from dev_double.client import DevDouble


def test_documented_import_path_still_works():
    assert DevDouble is AppsDevDouble


def test_client_calls_every_endpoint():
    with TestClient(create_app(Settings(engine="mock"))) as http, DevDouble() as sd:
        sd._http.close()
        sd._http = http  # TestClient is an httpx.Client
        assert sd.route("Reformat a date")["route"] in {"small", "medium", "large"}
        assert "allowed" in sd.guard("hello", scope="Airline support.")
        assert sd.gate("list_issues", {"repo": "x"}, context="List issues.")["decision"] in {"allow", "ask", "deny"}
        assert len(sd.classify({"a": "apples", "b": "bananas"}, inputs=["apples", "bananas"])["results"]) == 2
        assert 0 <= sd.judge("4", input="2+2?")["score"] <= 4
        assert len(sd.rerank("q", ["a", "b"], top_n=1)["results"]) == 1
        r = sd.decide("x", {"q": {"type": "binary", "question": "?"}})
        assert r["answers"]["q"]["type"] == "binary"
        e = sd.extract("Vendor: Acme", {"vendor": {"type": "string"}, "paid": {"type": "boolean"}})
        assert e["values"]["vendor"] == "Acme" and "probability" in e["fields"]["paid"]
