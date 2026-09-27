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


async def test_extract_builds_one_record_tool_and_maps_arguments(fake_needle):
    from stunt_double.core.decision.i_extractor import IExtractor
    from stunt_double.core.decision.t_extract import TExtractField, TExtractRequest, TFieldValue

    assert isinstance(fake_needle, IExtractor)
    fields = {
        "vendor": TExtractField("string", "Company that issued the invoice."),
        "total": TExtractField("number"),
        "qty": TExtractField("integer"),
        "due date": TExtractField("string"),
        "currency": TExtractField("enum", options=["EUR", "USD"]),
        "paid": TExtractField("boolean"),
    }
    FakeNeedle.reply = {
        "suppressed_calls": [
            {"arguments": {"vendor": "Acme", "total": "1,200.50", "qty": 3, "currency": "EUR", "paid": False}}
        ],
        "confidence": 0.9,
    }
    t = Tracker(ClockMockImpl())
    req = TExtractRequest("Invoice from Acme", fields)
    got = await fake_needle.extract(req, t)
    assert got["vendor"] == TFieldValue("Acme", 0.9)
    assert got["total"] == TFieldValue(1200.5, 0.9) and got["qty"] == TFieldValue(3, 0.9)
    assert got["due date"] == TFieldValue(None, 0.9)
    assert got["currency"] == TFieldValue("EUR", 0.9, probabilities={"EUR": 0.9, "USD": 0.1})
    assert got["paid"] == TFieldValue(False, 0.9, probability=0.1)
    tool = FakeNeedle.instances[0].tools[0]
    schema = tool.model_json_schema(by_alias=True)
    assert set(schema["properties"]) == set(fields)
    assert schema["properties"]["vendor"]["description"] == "Company that issued the invoice."
    await fake_needle.extract(req, t)
    assert len(FakeNeedle.instances) == 1  # one agent per distinct field set
    assert t.warnings == []


async def test_extract_without_a_call_leaves_fields_empty(fake_needle):
    from stunt_double.core.decision.t_extract import TExtractField, TExtractRequest, TFieldValue

    FakeNeedle.reply = {"function_calls": [], "confidence": 0.3}
    t = Tracker(ClockMockImpl())
    got = await fake_needle.extract(TExtractRequest("x", {"a": TExtractField("string")}), t)
    assert got == {"a": TFieldValue(None, 0.0)}
    assert t.warnings == ["needle made no valid call"]


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


@pytest.mark.skipif(not has_module("needle"), reason="cactus-needle is not installed")
async def test_needle_extracts_a_record():
    from stunt_double.core.decision.t_extract import TExtractField, TExtractRequest
    from stunt_double.providers.needle.decision.decider_needle_impl import DeciderNeedleImpl

    d = DeciderNeedleImpl()
    req = TExtractRequest(
        "Invoice from Acme Corp. Amount due: 1200 EUR by 2026-10-01.",
        {
            "vendor": TExtractField("string", "Company that issued the invoice."),
            "total": TExtractField("number", "Amount due."),
            "currency": TExtractField("enum", options=["EUR", "USD", "GBP"]),
        },
    )
    try:
        got = await timed("needle extract", lambda: d.extract(req, Tracker(ClockMockImpl())))
    finally:
        await d.aclose()
    assert set(got) == set(req.fields)
    assert all(0 <= v.confidence <= 1 for v in got.values())
