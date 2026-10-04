"""POST /v1/systemone against the mock engine: Ollama's contract, plus `meta`."""

import json

import httpx
import pytest
from _shared import images
from fastapi.testclient import TestClient

from dev_double.apps.config import Settings
from dev_double.apps.server.app import create_app
from dev_double.core.decision.decider_basic_impl import DeciderBasicImpl
from dev_double.core.decision.decision_service_basic_impl import DecisionServiceBasicImpl
from dev_double.core.decision.errors import EngineError
from dev_double.core.decision.i_decider import IDecider
from dev_double.core.decision.t_answer import TChoiceAnswer
from dev_double.providers.mock.decision.clock_mock_impl import ClockMockImpl
from dev_double.providers.mock.decision.engine_mock_impl import EngineMockImpl
from dev_double.providers.mock.decision.id_provider_mock_impl import IdProviderMockImpl
from dev_double.providers.openai.decision.engine_openai_impl import EngineOpenAIImpl

# The example request from Ollama's OpenAPI spec for /v1/systemone.
OFFICIAL_REQUEST = {
    "model": "nimble",
    "state": "Our checkout has returned 500 errors since 9am.",
    "questions": {
        "label": {
            "type": "choice",
            "instructions": "Which label fits this ticket?",
            "criteria": {
                "billing": "Payments and refunds",
                "bug": "Software errors",
                "account": "Login and account access",
            },
        }
    },
}

ALL_TYPES = {
    "model": "nimble",
    "state": "I was charged twice and I want my money back, this is the worst service ever.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
        "topic": {
            "type": "choice",
            "instructions": "What is the ticket about?",
            "criteria": {"billing": "charged twice, money, refund", "bug": "software errors", "account": None},
        },
        "anger": {
            "type": "score",
            "instructions": "How upset is the customer?",
            "criteria": ["Calm.", "Annoyed.", "Furious, the worst service ever."],
        },
    },
}

CHOICE_KEYS = {"type", "choice", "probabilities", "confidence"}
NOUL_KEYS = {"type", "noul"}
SCORE_KEYS = {"type", "score", "legend", "probabilities", "confidence"}


@pytest.fixture
def client():
    with TestClient(create_app(Settings(engine="mock"))) as c:
        yield c


def post(client, **overrides):
    return client.post("/v1/systemone", json={**OFFICIAL_REQUEST, **overrides})


# ------------------------------------------------------------------ the contract


def test_the_official_example_gets_a_contract_response(client):
    r = client.post("/v1/systemone", json=OFFICIAL_REQUEST)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"model", "answers", "usage", "meta"}
    assert body["model"] == "nimble"
    assert set(body["answers"]) == {"label"}
    answer = body["answers"]["label"]
    assert set(answer) == CHOICE_KEYS  # nothing extra inside answers
    assert answer["type"] == "choice" and answer["choice"] in {"billing", "bug", "account"}
    assert set(answer["probabilities"]) == {"billing", "bug", "account"}
    assert sum(answer["probabilities"].values()) == pytest.approx(1, abs=1e-3)
    assert answer["choice"] == max(answer["probabilities"], key=answer["probabilities"].get)
    assert 0 <= answer["confidence"] <= 1
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}
    assert body["usage"]["input_tokens"] > 0 and isinstance(body["usage"]["output_tokens"], int)


def test_every_answer_type_has_exactly_the_contract_keys(client):
    r = client.post("/v1/systemone", json=ALL_TYPES)
    assert r.status_code == 200, r.text
    body = r.json()
    answers = body["answers"]
    assert list(answers) == ["refund", "topic", "anger"]  # the request's order

    noul = answers["refund"]
    assert set(noul) == NOUL_KEYS and noul["type"] == "noul"
    assert isinstance(noul["noul"], float) and 0 <= noul["noul"] <= 1  # a number, not a Boolean

    topic = answers["topic"]
    assert set(topic) == CHOICE_KEYS
    assert set(topic["probabilities"]) == {"billing", "bug", "account"}
    assert sum(topic["probabilities"].values()) == pytest.approx(1, abs=1e-3)

    score = answers["anger"]
    assert set(score) == SCORE_KEYS and score["type"] == "score"
    assert set(score["probabilities"]) == {"0", "1", "2"}
    assert sum(score["probabilities"].values()) == pytest.approx(1, abs=1e-3)
    assert score["legend"] == {"0": "Calm.", "1": "Annoyed.", "2": "Furious, the worst service ever."}
    expected = sum(int(k) * p for k, p in score["probabilities"].items())
    assert score["score"] == pytest.approx(expected, abs=1e-3)  # probability-weighted index, not normalized
    assert 0 <= score["score"] <= 2


def test_meta_is_the_one_extra_field(client):
    meta = client.post("/v1/systemone", json=OFFICIAL_REQUEST).json()["meta"]
    assert set(meta) == {"engine", "model", "latency_ms", "usage", "warnings"}
    assert meta["engine"] == "mock" and meta["model"] == "mock-overlap"


def test_state_and_instructions_may_be_objects_or_arrays(client):
    r = post(
        client,
        state={"ticket": {"title": "Checkout 500"}, "tags": ["urgent"]},
        questions={"q": {"type": "noul", "instructions": {"check": "is this urgent?"}}},
    )
    assert r.status_code == 200, r.text
    assert set(r.json()["answers"]["q"]) == NOUL_KEYS
    assert post(client, state=["a", "b"]).status_code == 200


def test_noul_criteria_and_keep_alive_are_accepted(client):
    q = {"type": "noul", "instructions": "Refund?", "criteria": {"true": "wants money back", "false": "does not"}}
    for keep_alive in ("5m", 0, -1, 30.5):
        assert post(client, questions={"q": q}, keep_alive=keep_alive).status_code == 200
    assert post(client, questions={"q": {**q, "criteria": {}}}).status_code == 200


def test_the_contract_limits_are_accepted(client):
    many = {f"option{i}": f"d{i}" for i in range(26)}
    q = {"type": "choice", "instructions": "?", "criteria": many}
    body = post(client, questions={"q": q}).json()
    assert len(body["answers"]["q"]["probabilities"]) == 26
    levels = {"type": "score", "instructions": "?", "criteria": [f"level {i}" for i in range(26)]}
    body = post(client, questions={"q": levels}).json()
    assert len(body["answers"]["q"]["legend"]) == 26 and 0 <= body["answers"]["q"]["score"] <= 25
    sixty_four = {f"q{i}": {"type": "noul", "instructions": f"Question {i}?"} for i in range(64)}
    assert len(post(client, questions=sixty_four).json()["answers"]) == 64


def test_an_empty_images_list_is_fine(client):
    assert post(client, images=[]).status_code == 200


# ------------------------------------------------------------------ errors are 400 {"error": ...}


def choice(criteria):
    return {"type": "choice", "instructions": "?", "criteria": criteria}


INVALID = [
    ("bad type", {"questions": {"q": {"type": "bad", "instructions": "?"}}}, "questions.q"),
    ("one option", {"questions": {"q": choice({"a": "only one"})}}, "2 to 26 entries, got 1"),
    ("27 options", {"questions": {"q": choice({f"o{i}": "x" for i in range(27)})}}, "2 to 26 entries, got 27"),
    ("similar keys", {"questions": {"q": choice({"yes": "a", "YES!": "b"})}}, "too similar"),
    ("one level", {"questions": {"q": {"type": "score", "instructions": "?", "criteria": ["one"]}}}, "2 to 26"),
    ("27 levels", {"questions": {"q": {"type": "score", "instructions": "?", "criteria": ["x"] * 27}}}, "got 27"),
    ("noul extra key", {"questions": {"q": {"type": "noul", "instructions": "?", "criteria": {"maybe": "x"}}}}, "maybe"),
    ("blank model", {"model": "   "}, "`model` must not be blank"),
    ("empty model", {"model": ""}, "`model` must not be blank"),
    ("empty questions", {"questions": {}}, "1 to 64 entries, got 0"),
    (
        "65 questions",
        {"questions": {f"q{i}": {"type": "noul", "instructions": "?"} for i in range(65)}},
        "1 to 64 entries, got 65",
    ),
    ("blank question name", {"questions": {" ": {"type": "noul", "instructions": "?"}}}, "must not be blank"),
    ("blank state", {"state": "  "}, "state"),
    ("numeric state", {"state": 5}, "state"),
    ("blank instructions", {"questions": {"q": {"type": "noul", "instructions": ""}}}, "must not be blank"),
    ("missing instructions", {"questions": {"q": {"type": "noul"}}}, "instructions"),
    ("images", {"images": ["aGVsbG8="]}, "images[0]"),
]


@pytest.mark.parametrize("name, overrides, message", INVALID, ids=[case[0] for case in INVALID])
def test_invalid_requests_are_400_with_an_error_message(client, name, overrides, message):
    r = post(client, **overrides)
    assert r.status_code == 400, r.text
    assert set(r.json()) == {"error"}
    assert message in r.json()["error"]


def test_missing_fields_and_malformed_bodies_are_400(client):
    for field in ("model", "state", "questions"):
        body = {k: v for k, v in OFFICIAL_REQUEST.items() if k != field}
        r = client.post("/v1/systemone", json=body)
        assert r.status_code == 400 and r.json()["error"].startswith(field), r.text
    r = client.post("/v1/systemone", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json() == {"error": "request body must be valid JSON"}
    assert client.post("/v1/systemone", json=["not", "an", "object"]).status_code == 400


def test_other_routes_keep_their_422(client):
    assert client.post("/v1/route", json={"input": "x", "routes": {"only": "one"}}).status_code == 422
    assert client.post("/v1/decide", json={"input": "x", "questions": {}}).status_code == 422


# ------------------------------------------------------------------ images


class Seeing(IDecider):
    """A decider that reads images, and records what it was given."""

    name, model = "seeing", "eyes"
    supports_images = True

    def __init__(self) -> None:
        self.images: list[tuple[str, ...]] = []

    def model_for(self, images):
        return "eyes-vision" if images else self.model

    async def ask(self, value, question, tracker, label="", images=()):
        self.images.append(images)
        options = list(question.options)
        return TChoiceAnswer(value=options[0], probabilities={o: 1 / len(options) for o in options}, confidence=0.0)

    async def aclose(self) -> None:
        return None


@pytest.fixture
def seeing():
    decider = Seeing()
    with TestClient(create_app(Settings(engine="mock"), decider=decider)) as c:
        yield c, decider


def test_images_reach_the_decider_in_request_order(seeing):
    c, decider = seeing
    r = post(c, images=[images.PNG, images.JPEG, images.WEBP])
    assert r.status_code == 200, r.text
    assert decider.images == [(images.PNG, images.JPEG, images.WEBP)]
    assert r.json()["meta"]["model"] == "eyes-vision"


def test_a_request_without_images_sends_none(seeing):
    c, decider = seeing
    assert post(c).status_code == 200
    assert post(c, images=[]).status_code == 200
    assert decider.images == [(), ()]


@pytest.mark.parametrize(
    "bad, message",
    [
        ("data:image/png;base64," + images.PNG, "not a data URL"),
        ("http://localhost/cat.png", "not a URL"),
        ("not base64 at all!", "not valid base64"),
        (images.b64(b"just some text, not a picture"), "not a PNG, JPEG, or WebP"),
        (images.b64(b"GIF89a" + b"\x00" * 20), "not a PNG, JPEG, or WebP"),
    ],
    ids=["data url", "url", "garbage", "text", "gif"],
)
def test_bad_images_are_400_even_on_an_engine_that_reads_images(seeing, bad, message):
    c, decider = seeing
    r = post(c, images=[images.PNG, bad])
    assert r.status_code == 400
    assert set(r.json()) == {"error"} and message in r.json()["error"] and "images[1]" in r.json()["error"]
    assert decider.images == []


def test_an_engine_without_vision_says_so_plainly(client):
    r = post(client, images=[images.PNG])
    assert r.status_code == 400
    assert r.json() == {"error": "this engine does not support images"}


def test_the_openai_engine_without_a_vision_model_says_how_to_set_one():
    app = create_app(Settings(engine="openai"), engine=openai_engine_that_must_not_be_called())
    with TestClient(app) as c:
        r = post(c, images=[images.PNG])
    assert r.status_code == 400
    assert r.json()["error"].startswith("this engine does not support images")
    assert "--vision-model" in r.json()["error"]


def test_the_openai_engine_with_a_vision_model_gets_the_images_and_reports_it():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        top = [{"token": "billing", "logprob": -0.1}, {"token": "bug", "logprob": -3.0}]
        choice = {"message": {"content": "billing"}, "logprobs": {"content": [{"token": "billing", "top_logprobs": top}]}}
        return httpx.Response(200, json={"choices": [choice], "usage": {"prompt_tokens": 1000, "completion_tokens": 1}})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    engine = EngineOpenAIImpl("http://model/v1", "text-model", client=http, vision_model="vision-model")
    with TestClient(create_app(Settings(engine="openai"), engine=engine)) as c:
        r = post(c, images=[images.PNG])
    assert r.status_code == 200, r.text
    assert r.json()["meta"]["model"] == "vision-model"
    assert r.json()["answers"]["label"]["choice"] == "billing"
    assert seen[0]["model"] == "vision-model"
    first = seen[0]["messages"][1]["content"][0]
    assert first == {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images.PNG}"}}


# ------------------------------------------------------------------ 413


def test_a_body_over_64_kib_without_images_is_413(client):
    r = post(client, state="x" * (64 * 1024))
    assert r.status_code == 413
    assert r.json() == {"error": "request body must not exceed 64 KiB without images"}


def test_a_body_just_under_64_kib_is_fine(client):
    assert post(client, state="x" * (60 * 1024)).status_code == 200


def test_a_large_body_with_images_passes_the_size_check(client):
    big = images.b64(b"\x89PNG\r\n\x1a\n" + b"\x00" * (100 * 1024))
    r = post(client, images=[big])
    assert r.status_code == 400  # past the 413, then refused as unsupported
    assert r.json() == {"error": "this engine does not support images"}


def test_a_body_over_32_mib_with_images_is_413(client):
    r = post(client, images=["A" * (33 * 1024 * 1024)])
    assert r.status_code == 413
    assert r.json() == {"error": "request body must not exceed 32 MiB with images"}


# ------------------------------------------------------------------ model handling


def test_the_request_model_is_echoed_but_the_configured_model_answers(client):
    body = client.post("/v1/systemone", json=OFFICIAL_REQUEST).json()
    assert body["model"] == "nimble"
    assert body["meta"]["model"] == "mock-overlap"
    [warning] = body["meta"]["warnings"]
    assert "'nimble'" in warning and "'mock-overlap'" in warning and "--honor-request-model" in warning


def test_no_warning_when_the_request_names_the_configured_model(client):
    body = post(client, model="mock-overlap").json()
    assert body["model"] == "mock-overlap" and body["meta"]["warnings"] == []


class NamedMock(EngineMockImpl):
    """A mock engine reporting a given model name."""

    def __init__(self, model: str) -> None:
        self.model = model


def named_service(settings: Settings) -> DecisionServiceBasicImpl:
    return DecisionServiceBasicImpl(DeciderBasicImpl(NamedMock(settings.model)), ClockMockImpl(), IdProviderMockImpl())


@pytest.fixture
def built(monkeypatch):
    """Records the settings each per-model service is built from."""
    settings_seen: list[Settings] = []

    def build(settings: Settings) -> DecisionServiceBasicImpl:
        settings_seen.append(settings)
        return named_service(settings)

    monkeypatch.setattr("dev_double.apps.server.model_services.build_service", build)
    return settings_seen


def honoring(engine: str = "openai") -> TestClient:
    return TestClient(create_app(Settings(engine=engine, model="configured", honor_request_model=True)))


def test_honor_request_model_answers_with_the_requested_model(built):
    with honoring() as c:
        body = c.post("/v1/systemone", json=OFFICIAL_REQUEST).json()
    assert body["model"] == "nimble"
    assert body["meta"]["model"] == "nimble"  # the real model that answered
    assert body["meta"]["warnings"] == []
    assert [s.model for s in built] == ["nimble"] and built[0].engine == "openai"


def test_honor_request_model_replaces_the_text_model_and_keeps_the_vision_model(built):
    settings = Settings(engine="openai", model="configured", vision_model="seer", honor_request_model=True)
    with TestClient(create_app(settings)) as c:
        c.post("/v1/systemone", json={**OFFICIAL_REQUEST, "images": [images.PNG]})
    assert [(s.model, s.vision_model) for s in built] == [("nimble", "seer")]


def test_honor_request_model_builds_one_service_per_model(built):
    with honoring("systemone") as c:
        for model in ("a", "b", "a", "a"):
            assert c.post("/v1/systemone", json={**OFFICIAL_REQUEST, "model": model}).status_code == 200
    assert [(s.engine, s.model) for s in built] == [("systemone", "a"), ("systemone", "b")]


def test_honor_request_model_with_the_configured_model_uses_the_default(built):
    with honoring() as c:
        # The default service talks to a real (absent) model server, so only check which one is chosen.
        assert c.app.state.models.honors("configured") is False
        assert c.app.state.models.honors("other") is True
        assert c.app.state.models.honors("  ") is False and c.app.state.models.honors(None) is False


def test_honor_request_model_is_ignored_with_a_warning_by_mock_and_needle_engines(built):
    with honoring("mock") as c:
        body = c.post("/v1/systemone", json=OFFICIAL_REQUEST).json()
    assert body["model"] == "nimble" and body["meta"]["model"] == "mock-overlap"
    [warning] = body["meta"]["warnings"]
    assert "can't be selected on this engine" in warning and "'nimble'" in warning
    assert built == []


def test_honor_request_model_with_an_injected_engine_is_ignored_with_a_warning(built):
    app = create_app(Settings(engine="openai", honor_request_model=True), engine=EngineMockImpl())
    with TestClient(app) as c:
        body = c.post("/v1/systemone", json=OFFICIAL_REQUEST).json()
    assert body["meta"]["model"] == "mock-overlap" and len(body["meta"]["warnings"]) == 1 and built == []


def test_honor_request_model_applies_to_every_endpoint_with_a_model_field(built):
    with honoring() as c:
        decided = c.post(
            "/v1/decide",
            json={"input": "x", "model": "m1", "questions": {"q": {"type": "binary", "question": "?"}}},
        ).json()
        reranked = c.post("/v1/rerank", json={"query": "q", "documents": ["a", "b"], "model": "m2"}).json()
        no_model = c.get("/health").json()
    assert decided["meta"]["model"] == "m1" and reranked["meta"]["model"] == "m2"
    assert reranked["meta"]["warnings"] == []
    assert no_model["model"] == "configured"
    assert [s.model for s in built] == ["m1", "m2"]


def test_without_honor_request_model_decide_and_rerank_ignore_their_model(client):
    reranked = client.post("/v1/rerank", json={"query": "q", "documents": ["a"], "model": "m2"}).json()
    assert reranked["meta"]["model"] == "mock-overlap" and reranked["meta"]["warnings"] == []


def test_honor_request_model_on_a_mock_engine_warns_decide_and_rerank_too(built):
    with honoring("mock") as c:
        reranked = c.post("/v1/rerank", json={"query": "q", "documents": ["a"], "model": "m2"}).json()
    assert len(reranked["meta"]["warnings"]) == 1 and "'m2'" in reranked["meta"]["warnings"][0]


def test_honor_request_model_setting_from_env(monkeypatch):
    assert Settings().honor_request_model is False
    monkeypatch.setenv("DEV_DOUBLE_HONOR_REQUEST_MODEL", "1")
    assert Settings.from_env().honor_request_model is True
    monkeypatch.setenv("DEV_DOUBLE_HONOR_REQUEST_MODEL", "false")
    assert Settings.from_env().honor_request_model is False


# ------------------------------------------------------------------ engine failures


def test_engine_failure_is_500_with_an_error_message():
    class Down(EngineMockImpl):
        async def distribution(self, query):
            raise EngineError("model server is down")

    with TestClient(create_app(Settings(engine="mock"), engine=Down())) as c:
        r = c.post("/v1/systemone", json=OFFICIAL_REQUEST)
    assert r.status_code == 500
    assert r.json() == {"error": "model server is down"}


def openai_engine_that_must_not_be_called() -> EngineOpenAIImpl:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("the model server should not be called")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return EngineOpenAIImpl("http://model/v1", "tiny", client=client)


def test_more_than_20_options_on_the_openai_engine_is_a_400():
    app = create_app(Settings(engine="openai"), engine=openai_engine_that_must_not_be_called())
    many = {f"option{i}": f"d{i}" for i in range(26)}
    with TestClient(app) as c:
        r = post(c, questions={"q": choice(many)})
        assert r.status_code == 400
        assert r.json() == {"error": "this engine supports at most 20 options; got 26"}
        # Other routes report the same condition as a client error, in their own error shape.
        r = c.post("/v1/classify", json={"input": "x", "labels": many})
        assert r.status_code == 400
        assert r.json() == {
            "error": "unsupported_request",
            "detail": "this engine supports at most 20 options; got 26",
        }


# ------------------------------------------------------------------ OpenAPI


def test_openapi_describes_the_endpoint(client):
    spec = client.get("/openapi.json").json()
    op = spec["paths"]["/v1/systemone"]["post"]
    assert set(op["responses"]) >= {"200", "400", "413", "500"}
    schemas = spec["components"]["schemas"]
    assert {"SystemOneRequest", "SystemOneResponse", "SystemOneChoiceQuestion", "SystemOneNoulAnswer"} <= set(schemas)
    assert set(schemas["SystemOneResponse"]["required"]) == {"model", "answers", "usage", "meta"}
    assert "deprecated" not in op


def test_decide_is_deprecated(client):
    spec = client.get("/openapi.json").json()
    assert spec["paths"]["/v1/decide"]["post"]["deprecated"] is True
    assert "deprecated" not in spec["paths"]["/v1/route"]["post"]


DECIDE = {"input": "I want a refund.", "questions": {"q": {"type": "binary", "question": "Refund?"}}}


def test_decide_still_works_but_says_it_is_deprecated(client):
    r = client.post("/v1/decide", json=DECIDE)
    assert r.status_code == 200
    assert r.headers["deprecation"] == "true"
    body = r.json()
    assert body["answers"]["q"]["type"] == "binary"
    assert "/v1/decide is deprecated and will be removed in 0.3; use /v1/systemone" in body["meta"]["warnings"]


def test_systemone_is_not_marked_deprecated(client):
    r = client.post("/v1/systemone", json=OFFICIAL_REQUEST)
    assert "deprecation" not in r.headers
    assert all("deprecated" not in w for w in r.json()["meta"]["warnings"])


def test_recording_captures_systemone_calls(tmp_path):
    path = tmp_path / "calls.jsonl"
    with TestClient(create_app(Settings(engine="mock", record=str(path)))) as c:
        c.post("/v1/systemone", json=OFFICIAL_REQUEST)
    entry = json.loads(path.read_text(encoding="utf-8").strip())
    assert entry["path"] == "/v1/systemone" and entry["status"] == 200
    assert entry["request"]["model"] == "nimble" and "answers" in entry["response"]
