import pytest

from stunt_double.core.decision.decider_basic_impl import DeciderBasicImpl
from stunt_double.core.decision.errors import EngineError
from stunt_double.core.decision.t_answer import TBinaryAnswer, TChoiceAnswer, TScaleAnswer
from stunt_double.core.decision.t_label_query import TLabelQuery, TLabelResult
from stunt_double.core.decision.t_question import TBinaryQuestion, TChoiceQuestion, TScaleQuestion
from stunt_double.core.decision.t_usage import TUsage
from stunt_double.core.decision.tracker import Tracker
from stunt_double.providers.mock.decision.clock_mock_impl import ClockMockImpl
from stunt_double.providers.mock.decision.engine_mock_impl import EngineMockImpl


class ScriptedEngine:
    """Returns a fixed result and remembers the queries it saw."""

    name = "scripted"
    model = "s1"

    def __init__(self, result: TLabelResult) -> None:
        self.result = result
        self.queries: list[TLabelQuery] = []
        self.closed = False

    async def distribution(self, query: TLabelQuery) -> TLabelResult:
        self.queries.append(query)
        return self.result

    async def aclose(self) -> None:
        self.closed = True


async def test_mock_engine_answers_all_question_types():
    d = DeciderBasicImpl(EngineMockImpl())
    t = Tracker(ClockMockImpl())
    text = "I want a refund for my cancelled flight."
    yes = await d.ask(text, TBinaryQuestion("Does the customer want a refund?"), t)
    pick = await d.ask(
        text,
        TChoiceQuestion(
            "What does the customer want?",
            {"refund": "Wants a refund of money.", "rebooking": "Wants another seat on a later departure."},
        ),
        t,
    )
    anger = await d.ask(text, TScaleQuestion("How upset?", ["Calm.", "Annoyed.", "Furious."]), t)
    assert isinstance(yes, TBinaryAnswer) and 0 <= yes.probability <= 1
    assert isinstance(pick, TChoiceAnswer) and pick.value == "refund"
    assert isinstance(anger, TScaleAnswer) and set(anger.probabilities) == {"0", "1", "2"}
    assert t.usage.input_tokens > 0 and t.usage.output_tokens == 3
    assert (d.name, d.model) == ("mock", "mock-overlap")


async def test_prompt_and_labels_reach_the_engine_and_warnings_are_labelled():
    engine = ScriptedEngine(TLabelResult({"yes": 0.9, "no": 0.1}, TUsage(10, 1), ["low mass"]))
    d = DeciderBasicImpl(engine)
    t = Tracker(ClockMockImpl())
    a = await d.ask({"k": "v"}, TBinaryQuestion("Is it?", yes="when so"), t, label="q1")
    assert a == TBinaryAnswer(value=True, probability=0.9, confidence=0.8)
    q = engine.queries[0]
    assert q.labels == ["yes", "no"]
    assert '"k": "v"' in q.user and 'Answer "yes" if: when so' in q.user
    assert t.warnings == ["q1: low mass"]
    assert (t.usage.input_tokens, t.usage.output_tokens) == (10, 1)
    await d.aclose()
    assert engine.closed


async def test_engine_errors_propagate():
    class Failing(ScriptedEngine):
        async def distribution(self, query: TLabelQuery) -> TLabelResult:
            raise EngineError("down")

    d = DeciderBasicImpl(Failing(TLabelResult({})))
    with pytest.raises(EngineError):
        await d.ask("x", TBinaryQuestion("?"), Tracker(ClockMockImpl()))
