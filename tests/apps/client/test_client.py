import httpx
import pytest
from _shared import images
from _shared.images import solid_png
from fastapi.testclient import TestClient

from dev_double.apps.client import DevDouble as AppsDevDouble
from dev_double.apps.config import Settings
from dev_double.apps.server.app import create_app
from dev_double.client import DevDouble, image_base64


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
        with pytest.warns(DeprecationWarning, match="systemone"):
            r = sd.decide("x", {"q": {"type": "binary", "question": "?"}})
        assert r["answers"]["q"]["type"] == "binary"
        s = sd.systemone(
            "nimble",
            "Our checkout has returned 500 errors since 9am.",
            {"q": {"type": "noul", "instructions": "Is something broken?"}},
            keep_alive="5m",
        )
        assert s["model"] == "nimble" and set(s["answers"]["q"]) == {"type", "noul"} and "meta" in s
        e = sd.extract("Vendor: Acme", {"vendor": {"type": "string"}, "paid": {"type": "boolean"}})
        assert e["values"]["vendor"] == "Acme" and "probability" in e["fields"]["paid"]


def test_systemone_leaves_out_unset_options():
    seen: list[dict] = []

    class Capture:
        def post(self, path, json):
            seen.append({"path": path, **json})
            raise RuntimeError("stop")

        def close(self):
            return None

    sd = DevDouble()
    sd._http = Capture()
    with pytest.raises(RuntimeError):
        sd.systemone("m", "state", {"q": {"type": "noul", "instructions": "?"}})
    assert seen == [
        {"path": "/v1/systemone", "model": "m", "state": "state", "questions": {"q": {"type": "noul", "instructions": "?"}}}
    ]


def test_systemone_sends_images_and_image_base64_reads_a_file(tmp_path):
    seen: list[dict] = []

    class Capture:
        def post(self, path, json):
            seen.append(json)
            raise RuntimeError("stop")

        def close(self):
            return None

    photo = tmp_path / "red.png"
    photo.write_bytes(solid_png((255, 0, 0), size=4))
    assert image_base64(photo) == images.PNG and image_base64(str(photo)) == images.PNG
    sd = DevDouble()
    sd._http = Capture()
    with pytest.raises(RuntimeError):
        sd.systemone("m", "state", {"q": {"type": "noul", "instructions": "?"}}, images=[image_base64(photo)])
    assert seen[0]["images"] == [images.PNG]


def test_systemone_with_images_end_to_end():
    with TestClient(create_app(Settings(engine="mock"))) as http, DevDouble() as sd:
        sd._http.close()
        sd._http = http
        with pytest.raises(httpx.HTTPStatusError) as exc:  # the mock engine can't read images
            sd.systemone("m", "s", {"q": {"type": "noul", "instructions": "?"}}, images=[images.PNG])
        assert exc.value.response.json() == {"error": "this engine does not support images"}
