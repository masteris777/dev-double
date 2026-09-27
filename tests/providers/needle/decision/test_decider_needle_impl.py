import pytest
from _shared.resources import has_module
from _shared.timing import timed

from stunt_double.core.decision.t_answer import TBinaryAnswer, TChoiceAnswer
from stunt_double.core.decision.t_question import TBinaryQuestion, TChoiceQuestion
from stunt_double.core.decision.tracker import Tracker
from stunt_double.providers.mock.decision.clock_mock_impl import ClockMockImpl


class FakeNeedle:
    """Stands in for needle.Needle: answers with a scripted tool call or embedding."""

    reply: dict = {}
    instances: list["FakeNeedle"] = []

    def __init__(self, tools=None, system=None, stateless=True):  # type: ignore[no-untyped-def]
        self.tools, self.system, self.closed = tools, system, False
        FakeNeedle.instances.append(self)

    def complete(self, text: str) -> dict:
        return FakeNeedle.reply

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0] if "password" in text else [0.0, 1.0]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_needle(monkeypatch):
    import types

    monkeypatch.setitem(__import__("sys").modules, "needle", types.SimpleNamespace(Needle=FakeNeedle))
    FakeNeedle.instances = []
    from stunt_double.providers.needle.decision.decider_needle_impl import DeciderNeedleImpl

    return DeciderNeedleImpl()


async def test_tool_call_confidence_becomes_a_distribution(fake_needle):
    FakeNeedle.reply = {"function_calls": [{"arguments": {"label": "rebooking"}}], "confidence": 0.8}
    q = TChoiceQuestion("Which?", {"refund": "Money.", "rebooking": "Flight.", "information": "Info."})
    a = await fake_needle.ask("x", q, Tracker(ClockMockImpl()))
    assert isinstance(a, TChoiceAnswer)
    assert a.value == "rebooking" and a.probabilities == {"refund": 0.1, "rebooking": 0.8, "information": 0.1}
    assert "Allowed labels: refund: Money.; rebooking: Flight.; information: Info." in FakeNeedle.instances[0].system


async def test_no_valid_call_is_uniform_with_warning(fake_needle):
    FakeNeedle.reply = {"function_calls": []}
    t = Tracker(ClockMockImpl())
    a = await fake_needle.ask("x", TBinaryQuestion("?"), t, label="documents[0]")
    assert a == TBinaryAnswer(value=True, probability=0.5, confidence=0.0)
    assert t.warnings == ["documents[0]: needle made no valid call"]


async def test_rerank_uses_embeddings_and_close_closes_agents(fake_needle):
    from stunt_double.core.decision.i_reranker import IReranker

    assert isinstance(fake_needle, IReranker)
    scores = await fake_needle.rerank_scores("password?", ["office hours", "reset password"], Tracker(ClockMockImpl()))
    assert scores == [0.0, 1.0]
    await fake_needle.aclose()
    assert all(n.closed for n in FakeNeedle.instances)


@pytest.mark.skipif(not has_module("needle"), reason="cactus-needle is not installed")
async def test_needle_answers_and_reranks():
    from stunt_double.providers.needle.decision.decider_needle_impl import DeciderNeedleImpl

    d = DeciderNeedleImpl()
    t = Tracker(ClockMockImpl())
    try:
        yes = await timed(
            "needle binary",
            lambda: d.ask(
                "The package arrived broken and I want my money back.",
                TBinaryQuestion("Is the customer asking for a refund?"),
                t,
            ),
        )
        pick = await timed(
            "needle choice",
            lambda: d.ask(
                "My flight was cancelled, please put me on the next one.",
                TChoiceQuestion(
                    "What does the customer want?",
                    {"refund": "Money returned.", "rebooking": "A replacement flight.", "information": "Only information."},
                ),
                t,
            ),
        )
        scores = await timed(
            "needle rerank",
            lambda: d.rerank_scores(
                "How do I reset my password?",
                ["Our office is closed on public holidays.", "To reset your password, open Settings > Security."],
                t,
            ),
        )
    finally:
        await d.aclose()
    assert isinstance(yes, TBinaryAnswer) and 0 <= yes.probability <= 1
    assert isinstance(pick, TChoiceAnswer) and sum(pick.probabilities.values()) == pytest.approx(1, abs=1e-3)
    assert scores[1] > scores[0]
