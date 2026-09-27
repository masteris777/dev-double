import json

import httpx
import pytest
from _shared.resources import SYSTEMONE_URL, has_systemone
from _shared.timing import timed

from stunt_double.core.decision.errors import EngineError
from stunt_double.core.decision.t_answer import TBinaryAnswer, TChoiceAnswer, TScaleAnswer
from stunt_double.core.decision.t_question import TBinaryQuestion, TChoiceQuestion, TScaleQuestion
from stunt_double.core.decision.tracker import Tracker
from stunt_double.providers.mock.decision.clock_mock_impl import ClockMockImpl
from stunt_double.providers.systemone.decision.decider_system_one_impl import DeciderSystemOneImpl


def decider_returning(answer: dict, status: int = 200, usage: dict | None = None):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = {"answers": {"q": answer}, "usage": usage or {"input_tokens": 7, "output_tokens": 1}}
        return httpx.Response(status, json=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return DeciderSystemOneImpl("kev", "http://s1/", api_key="k", client=client), seen


async def test_binary_request_and_answer_mapping():
    d, seen = decider_returning({"noul": 0.83})
    t = Tracker(ClockMockImpl())
    a = await d.ask({"ticket": "refund"}, TBinaryQuestion("Refund?", yes="asks for money"), t)
    assert a == TBinaryAnswer(value=True, probability=0.83, confidence=0.66)
    sent = seen[0]
    assert str(sent.url) == "http://s1/v1/systemone"
    assert sent.headers["authorization"] == "Bearer k"
    assert json.loads(sent.content) == {
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


# ------------------------------------------------------------------ integration


@pytest.mark.skipif(not has_systemone(), reason=f"no System 1 server at {SYSTEMONE_URL}/v1/systemone")
async def test_live_system_one_server():
    from stunt_double.providers.std.decision.clock_std_impl import ClockStdImpl

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
