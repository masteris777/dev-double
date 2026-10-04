"""The extract use case: enum/boolean fields as questions, text fields generated."""

import pytest

from dev_double.core.decision.decider_basic_impl import DeciderBasicImpl
from dev_double.core.decision.decision_service_basic_impl import DecisionServiceBasicImpl
from dev_double.core.decision.extract_questions import boolean_question, enum_question
from dev_double.core.decision.i_decider import IDecider
from dev_double.core.decision.t_answer import TBinaryAnswer, TChoiceAnswer
from dev_double.core.decision.t_extract import TExtractField, TExtractRequest, TFieldValue
from dev_double.core.decision.t_question import TBinaryQuestion, TChoiceQuestion
from dev_double.providers.mock.decision.clock_mock_impl import ClockMockImpl
from dev_double.providers.mock.decision.engine_mock_impl import EngineMockImpl
from dev_double.providers.mock.decision.id_provider_mock_impl import IdProviderMockImpl

INVOICE = "INVOICE\nVendor: Acme Corp\nTotal: 1,200.50\nCurrency: euro, paid in euros\nStatus: already paid"

FIELDS = {
    "vendor": TExtractField("string", "Company that issued the invoice."),
    "total": TExtractField("number", "Amount due, without currency symbol."),
    "po_number": TExtractField("integer", "Purchase order number."),
    "currency": TExtractField("enum", options={"EUR": "euro euros", "USD": "US dollars", "GBP": "British pounds"}),
    "paid": TExtractField("boolean", "Whether the invoice is already paid."),
}


def service(decider) -> DecisionServiceBasicImpl:
    return DecisionServiceBasicImpl(decider, ClockMockImpl(step=0.25), IdProviderMockImpl())


def test_enum_and_boolean_questions():
    q = enum_question("currency", TExtractField("enum", "ISO code.", ["EUR", "USD"]))
    assert q == TChoiceQuestion(
        'What is the value of the field "currency" in the input? ISO code.', {"EUR": "EUR", "USD": "USD"}
    )
    assert enum_question("c", TExtractField("enum", options=["a", "b"])).question.endswith("input?")
    assert boolean_question("paid", TExtractField("boolean", "Already paid.")) == TBinaryQuestion(
        "Is this true for the input? Already paid."
    )
    assert boolean_question("paid", TExtractField("boolean")).question == "Is this true for the input? paid"


async def test_extract_with_the_mock_engine():
    r = await service(DeciderBasicImpl(EngineMockImpl())).extract(TExtractRequest(INVOICE, FIELDS))
    assert list(r.fields) == list(FIELDS)
    assert r.values["vendor"] == "Acme Corp" and r.values["total"] == 1200.5 and r.values["po_number"] is None
    cur = r.fields["currency"]
    assert cur.value == "EUR" and set(cur.probabilities) == {"EUR", "USD", "GBP"}
    assert sum(cur.probabilities.values()) == pytest.approx(1, abs=1e-3) and cur.probability is None
    paid = r.fields["paid"]
    assert isinstance(paid.value, bool) and 0 <= paid.probability <= 1 and paid.probabilities is None
    assert r.fields["vendor"].probabilities is None and r.fields["vendor"].probability is None
    assert (r.meta.engine, r.meta.latency_ms) == ("mock", 250)
    assert r.meta.usage.input_tokens > 0


class QuestionsOnly(IDecider):
    """A decider with no text generation (like a System 1 model)."""

    name, model = "q", "q1"

    async def ask(self, value, question, tracker, label="", images=()):
        if isinstance(question, TBinaryQuestion):
            return TBinaryAnswer(value=False, probability=0.1, confidence=0.8)
        return TChoiceAnswer(value="EUR", probabilities={"EUR": 0.97, "USD": 0.02, "GBP": 0.01}, confidence=0.955)

    async def aclose(self) -> None:
        return None


async def test_decider_without_field_reader_leaves_text_fields_empty():
    r = await service(QuestionsOnly()).extract(TExtractRequest(INVOICE, FIELDS))
    assert r.fields["currency"] == TFieldValue("EUR", 0.955, probabilities={"EUR": 0.97, "USD": 0.02, "GBP": 0.01})
    assert r.fields["paid"] == TFieldValue(False, 0.8, probability=0.1)
    assert r.fields["vendor"] == TFieldValue(None, 0.0)
    assert r.values["total"] is None
    assert "engine can't generate text: field vendor left empty" in r.meta.warnings
    assert len(r.meta.warnings) == 3


class NativeExtractor(QuestionsOnly):
    async def ask(self, value, question, tracker, label=""):  # pragma: no cover - not used
        raise AssertionError("extract should use the native extractor")

    async def extract(self, req, tracker):
        tracker.warn("native")
        return {"vendor": TFieldValue("Acme", 0.6)}


async def test_native_extractor_is_preferred_and_missing_fields_are_filled():
    r = await service(NativeExtractor()).extract(TExtractRequest(INVOICE, FIELDS))
    assert r.values == {"vendor": "Acme", "total": None, "po_number": None, "currency": None, "paid": None}
    assert r.fields["total"] == TFieldValue(None, 0.0)
    assert r.meta.warnings == ["native"]


class CountingMock(EngineMockImpl):
    def __init__(self) -> None:
        self.generated: list = []

    async def generate(self, query):
        self.generated.append(query)
        return await super().generate(query)


async def test_all_text_fields_share_one_generation():
    engine = CountingMock()
    r = await service(DeciderBasicImpl(engine)).extract(TExtractRequest(INVOICE, FIELDS))
    assert len(engine.generated) == 1
    assert engine.generated[0].json is True
    assert engine.generated[0].fields == ("vendor", "total", "po_number")  # enum/boolean are questions
    assert r.values["vendor"] == "Acme Corp"


async def test_only_enum_and_boolean_fields_need_no_generation():
    engine = CountingMock()
    fields = {k: FIELDS[k] for k in ("currency", "paid")}
    r = await service(DeciderBasicImpl(engine)).extract(TExtractRequest(INVOICE, fields))
    assert engine.generated == [] and r.values["currency"] == "EUR"
