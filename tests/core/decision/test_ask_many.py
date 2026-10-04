import asyncio

from dev_double.core.decision.decider_basic_impl import DeciderBasicImpl
from dev_double.core.decision.decision_service_basic_impl import DecisionServiceBasicImpl
from dev_double.core.decision.i_decider import IDecider
from dev_double.core.decision.t_answer import TBinaryAnswer
from dev_double.core.decision.t_decide import TDecideRequest
from dev_double.core.decision.t_guard import TGuardRequest
from dev_double.core.decision.t_question import TBinaryQuestion
from dev_double.core.decision.tracker import Tracker
from dev_double.providers.mock.decision.clock_mock_impl import ClockMockImpl
from dev_double.providers.mock.decision.engine_mock_impl import EngineMockImpl
from dev_double.providers.mock.decision.id_provider_mock_impl import IdProviderMockImpl
from dev_double.providers.needle.decision.decider_needle_impl import DeciderNeedleImpl
from dev_double.providers.systemone.decision.decider_system_one_impl import DeciderSystemOneImpl

QUESTIONS = {name: TBinaryQuestion(f"{name}?") for name in ("a", "b", "c")}


class Slow(IDecider):
    """Answers each question after a pause, and records how many ran at once."""

    name, model = "slow", "s1"

    def __init__(self) -> None:
        self.running = self.peak = 0
        self.calls: list[tuple[str, str]] = []

    async def ask(self, value, question, tracker, label="", images=()):
        self.calls.append((question.question, label))
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0.01)
        self.running -= 1
        tracker.warn("careful", label)
        return TBinaryAnswer(value=True, probability=0.9, confidence=0.8)

    async def aclose(self) -> None:
        return None


async def test_default_ask_many_asks_concurrently_and_keeps_order():
    d = Slow()
    t = Tracker(ClockMockImpl())
    answers = await d.ask_many("x", QUESTIONS, t)
    assert list(answers) == ["a", "b", "c"]
    assert d.peak == 3
    assert t.warnings == ["a: careful", "b: careful", "c: careful"]


async def test_ask_many_label_prefixes_the_question_names():
    d = Slow()
    t = Tracker(ClockMockImpl())
    await d.ask_many("x", {"a": QUESTIONS["a"]}, t, label="policies")
    assert d.calls == [("a?", "policies.a")]


def test_every_decider_inherits_the_default_ask_many():
    assert DeciderBasicImpl.ask_many is IDecider.ask_many
    assert DeciderNeedleImpl.ask_many is IDecider.ask_many
    assert DeciderSystemOneImpl.ask_many is not IDecider.ask_many  # one call for all questions


async def test_basic_decider_answers_many_questions():
    d = DeciderBasicImpl(EngineMockImpl())
    answers = await d.ask_many("a b c", QUESTIONS, Tracker(ClockMockImpl()))
    assert set(answers) == {"a", "b", "c"}


class Recording(Slow):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[str]] = []

    async def ask_many(self, value, questions, tracker, label="", images=()):
        self.batches.append(list(questions))
        return await super().ask_many(value, questions, tracker, label, images)


async def test_decide_and_guard_ask_their_questions_as_one_batch():
    d = Recording()
    svc = DecisionServiceBasicImpl(d, ClockMockImpl(), IdProviderMockImpl())
    await svc.decide(TDecideRequest("x", QUESTIONS))
    await svc.guard(TGuardRequest("x", policies={"p1": "one", "p2": "two"}))
    assert d.batches == [["a", "b", "c"], ["p1", "p2"]]


async def test_default_ask_many_passes_the_images_to_every_ask():
    class Seeing(Slow):
        async def ask(self, value, question, tracker, label="", images=()):
            self.seen_images.append(images)
            return await super().ask(value, question, tracker, label, images)

    d = Seeing()
    d.seen_images = []
    await d.ask_many("x", QUESTIONS, Tracker(ClockMockImpl()), images=("img",))
    assert d.seen_images == [("img",)] * 3


async def test_decide_hands_the_images_to_the_decider_and_reports_the_vision_model():
    class Seeing(Slow):
        supports_images = True

        def model_for(self, images):
            return "vision" if images else self.model

        async def ask_many(self, value, questions, tracker, label="", images=()):
            self.images = images
            return await super().ask_many(value, questions, tracker, label, images)

    d = Seeing()
    svc = DecisionServiceBasicImpl(d, ClockMockImpl(), IdProviderMockImpl())
    assert svc.supports_images is True
    assert (await svc.decide(TDecideRequest("x", QUESTIONS, images=("img",)))).meta.model == "vision"
    assert d.images == ("img",)
    assert (await svc.decide(TDecideRequest("x", QUESTIONS))).meta.model == "s1"


def test_deciders_do_not_read_images_unless_they_say_so():
    assert IDecider.supports_images.fget(Slow()) is False
    assert DeciderBasicImpl(EngineMockImpl()).supports_images is False
    assert DeciderNeedleImpl.supports_images is IDecider.supports_images
    assert DeciderSystemOneImpl("s1", "http://s1").supports_images is True
