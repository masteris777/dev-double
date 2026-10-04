import json

import httpx
import pytest
from _shared import images
from _shared.resources import SYSTEMONE_URL, has_systemone
from _shared.timing import timed

from dev_double.core.decision.decision_service_basic_impl import DecisionServiceBasicImpl
from dev_double.core.decision.errors import EngineError
from dev_double.core.decision.t_answer import TBinaryAnswer, TChoiceAnswer, TScaleAnswer
from dev_double.core.decision.t_decide import TDecideRequest
from dev_double.core.decision.t_extract import TExtractField, TExtractRequest, TFieldValue
from dev_double.core.decision.t_question import TBinaryQuestion, TChoiceQuestion, TScaleQuestion
from dev_double.core.decision.tracker import Tracker
from dev_double.providers.mock.decision.clock_mock_impl import ClockMockImpl
from dev_double.providers.mock.decision.id_provider_mock_impl import IdProviderMockImpl
from dev_double.providers.systemone.decision.decider_system_one_impl import DeciderSystemOneImpl


def decider_returning(answer: dict, status: int = 200, usage: dict | None = None):
    """A decider whose server answers every question with ``answer``."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        names = json.loads(request.content)["questions"]
        body = {"answers": {name: answer for name in names}, "usage": usage or {"input_tokens": 7, "output_tokens": 1}}
        return httpx.Response(status, json=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return DeciderSystemOneImpl("kev", "http://s1/", api_key="k", client=client), seen


def yes_no_server():
    """A decider whose server answers noul questions with 0.8 and records the request bodies."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        answers = {name: {"type": "noul", "noul": 0.8} for name in body["questions"]}
        usage = {"input_tokens": 100, "output_tokens": 3}
        return httpx.Response(200, json={"model": body["model"], "answers": answers, "usage": usage})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return DeciderSystemOneImpl("kev", "http://s1/", client=client, model="nimble"), seen


async def test_binary_request_and_answer_mapping():
    d, seen = decider_returning({"noul": 0.83})
    t = Tracker(ClockMockImpl())
    a = await d.ask({"ticket": "refund"}, TBinaryQuestion("Refund?", yes="asks for money"), t)
    assert a == TBinaryAnswer(value=True, probability=0.83, confidence=0.66)
    sent = seen[0]
    assert str(sent.url) == "http://s1/v1/systemone"
    assert sent.headers["authorization"] == "Bearer k"
    assert json.loads(sent.content) == {
        "model": "kev",
        "state": {"ticket": "refund"},
        "questions": {"q": {"type": "noul", "instructions": "Refund?", "criteria": {"true": "asks for money"}}},
    }
    assert (t.usage.input_tokens, t.usage.output_tokens) == (7, 1)
    assert (d.name, d.model) == ("kev", "kev")


async def test_choice_passes_probabilities_through():
    d, seen = decider_returning({"probabilities": {"a": 0.123456, "b": 0.876544}})
    a = await d.ask("x", TChoiceQuestion("Which?", {"a": "A", "b": "B"}), Tracker(ClockMockImpl()))
    assert isinstance(a, TChoiceAnswer)
    assert a.value == "b" and a.probabilities == {"a": 0.123456, "b": 0.876544}
    q = json.loads(seen[0].content)["questions"]["q"]
    assert q == {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B"}}


async def test_scale_mapping():
    d, seen = decider_returning({"probabilities": {"0": 0.2, "1": 0.8}})
    a = await d.ask("x", TScaleQuestion("How?", ["low", "high"]), Tracker(ClockMockImpl()))
    assert isinstance(a, TScaleAnswer)
    assert a.level == 1 and a.value == pytest.approx(0.8) and a.legend == {"0": "low", "1": "high"}
    assert json.loads(seen[0].content)["questions"]["q"]["type"] == "score"


async def test_server_errors_become_engine_errors():
    d, _ = decider_returning({}, status=500)
    with pytest.raises(EngineError, match="500"):
        await d.ask("x", TBinaryQuestion("?"), Tracker(ClockMockImpl()))
    d, _ = decider_returning({"unexpected": 1})
    with pytest.raises(EngineError, match="Unexpected"):
        await d.ask("x", TBinaryQuestion("?"), Tracker(ClockMockImpl()))


async def test_the_model_is_sent_in_every_request_body():
    d, seen = yes_no_server()
    assert d.model == "nimble"
    await d.ask("x", TBinaryQuestion("?"), Tracker(ClockMockImpl()))
    assert seen[0]["model"] == "nimble"
    # Without a model, the engine name stands in for it.
    plain, plain_seen = decider_returning({"noul": 0.5})
    await plain.ask("x", TBinaryQuestion("?"), Tracker(ClockMockImpl()))
    assert json.loads(plain_seen[0].content)["model"] == "kev"


async def test_many_questions_go_in_one_call_and_usage_is_recorded_once():
    d, seen = yes_no_server()
    questions = {f"q{i}": TBinaryQuestion(f"Question {i}?") for i in range(5)}
    t = Tracker(ClockMockImpl())
    answers = await d.ask_many("shared state", questions, t)
    assert len(seen) == 1
    assert list(seen[0]["questions"]) == list(questions)
    assert list(answers) == list(questions)
    assert all(a == TBinaryAnswer(value=True, probability=0.8, confidence=0.6) for a in answers.values())
    assert (t.usage.input_tokens, t.usage.output_tokens) == (100, 3)


async def test_more_than_64_questions_are_split_into_two_calls():
    d, seen = yes_no_server()
    questions = {f"q{i}": TBinaryQuestion(f"Question {i}?") for i in range(65)}
    t = Tracker(ClockMockImpl())
    answers = await d.ask_many("x", questions, t)
    assert sorted(len(body["questions"]) for body in seen) == [1, 64]
    assert list(answers) == list(questions)
    assert (t.usage.input_tokens, t.usage.output_tokens) == (200, 6)  # one usage per call


async def test_a_missing_answer_is_an_engine_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": {"a": {"type": "noul", "noul": 0.5}}, "usage": {}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    d = DeciderSystemOneImpl("kev", "http://s1", client=client)
    questions = {"a": TBinaryQuestion("A?"), "b": TBinaryQuestion("B?")}
    t = Tracker(ClockMockImpl())
    with pytest.raises(EngineError, match="no answer for question 'b'"):
        await d.ask_many("x", questions, t)
    assert t.usage.input_tokens == 0


async def test_an_answer_that_breaks_the_contract_is_an_engine_error():
    d, _ = decider_returning({"probabilities": {"a": 1.0}})  # no "b"
    with pytest.raises(EngineError, match="Unexpected"):
        await d.ask("x", TChoiceQuestion("Which?", {"a": "A", "b": "B"}), Tracker(ClockMockImpl()))


async def test_extract_asks_enum_and_boolean_fields_in_one_call():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        answers = {
            "currency": {"type": "choice", "choice": "EUR", "probabilities": {"EUR": 0.9, "USD": 0.1}, "confidence": 0.8},
            "paid": {"type": "noul", "noul": 0.25},
        }
        return httpx.Response(200, json={"answers": answers, "usage": {"input_tokens": 9, "output_tokens": 2}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    d = DeciderSystemOneImpl("kev", "http://s1/", client=client)
    svc = DecisionServiceBasicImpl(d, ClockMockImpl(), IdProviderMockImpl())
    fields = {
        "currency": TExtractField("enum", options=["EUR", "USD"]),
        "paid": TExtractField("boolean"),
        "vendor": TExtractField("string"),
    }
    r = await svc.extract(TExtractRequest("Invoice, 100 EUR", fields))
    assert r.fields["currency"] == TFieldValue("EUR", 0.8, probabilities={"EUR": 0.9, "USD": 0.1})
    assert r.fields["paid"].probability == 0.25
    assert r.fields["vendor"] == TFieldValue(None, 0.0)
    assert r.meta.warnings == ["engine can't generate text: field vendor left empty"]
    assert len(seen) == 1 and list(seen[0]["questions"]) == ["currency", "paid"]
    assert r.meta.usage.input_tokens == 9


async def test_decide_sends_all_its_questions_in_one_call():
    d, seen = yes_no_server()
    svc = DecisionServiceBasicImpl(d, ClockMockImpl(), IdProviderMockImpl())
    questions = {"one": TBinaryQuestion("One?"), "two": TBinaryQuestion("Two?")}
    r = await svc.decide(TDecideRequest("x", questions))
    assert len(seen) == 1 and list(r.answers) == ["one", "two"]
    assert r.meta.model == "nimble"


# ------------------------------------------------------------------ integration


@pytest.mark.skipif(not has_systemone(), reason=f"no System 1 server at {SYSTEMONE_URL}/v1/systemone")
async def test_live_system_one_server():
    from dev_double.providers.std.decision.clock_std_impl import ClockStdImpl

    d = DeciderSystemOneImpl("systemone", SYSTEMONE_URL)
    t = Tracker(ClockStdImpl())
    try:
        yes = await timed(
            "systemone binary",
            lambda: d.ask(
                "The package arrived broken and I want my money back.",
                TBinaryQuestion("Is the customer asking for a refund?"),
                t,
            ),
        )
        pick = await timed(
            "systemone choice",
            lambda: d.ask(
                "My flight was cancelled, please put me on the next one.",
                TChoiceQuestion(
                    "What does the customer want?",
                    {"refund": "Money returned.", "rebooking": "A replacement flight.", "information": "Only information."},
                ),
                t,
            ),
        )
    finally:
        await d.aclose()
    assert isinstance(yes, TBinaryAnswer) and 0 <= yes.probability <= 1
    assert isinstance(pick, TChoiceAnswer) and set(pick.probabilities) == {"refund", "rebooking", "information"}


async def test_images_are_sent_in_the_request_body_in_order():
    d, seen = yes_no_server()
    assert d.supports_images is True and d.model_for(True) == "nimble"
    questions = {"a": TBinaryQuestion("A?"), "b": TBinaryQuestion("B?")}
    await d.ask_many("x", questions, Tracker(ClockMockImpl()), images=(images.PNG, images.JPEG))
    await d.ask("x", TBinaryQuestion("C?"), Tracker(ClockMockImpl()), images=(images.WEBP,))
    assert [body["images"] for body in seen] == [[images.PNG, images.JPEG], [images.WEBP]]


async def test_no_images_key_without_images():
    d, seen = yes_no_server()
    await d.ask("x", TBinaryQuestion("?"), Tracker(ClockMockImpl()))
    assert "images" not in seen[0]


async def test_more_than_64_questions_send_the_images_in_every_call():
    d, seen = yes_no_server()
    questions = {f"q{i}": TBinaryQuestion(f"Question {i}?") for i in range(65)}
    await d.ask_many("x", questions, Tracker(ClockMockImpl()), images=(images.PNG,))
    assert [body["images"] for body in seen] == [[images.PNG]] * 2


async def test_the_servers_image_error_is_passed_through():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "this model does not support images"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    d = DeciderSystemOneImpl("kev", "http://s1", client=client)
    with pytest.raises(EngineError, match="400.*this model does not support images"):
        await d.ask("x", TBinaryQuestion("?"), Tracker(ClockMockImpl()), images=(images.PNG,))
