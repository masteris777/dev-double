import pytest

from dev_double.core.decision.t_label_query import TLabelQuery
from dev_double.providers.mock.decision.clock_mock_impl import ClockMockImpl
from dev_double.providers.mock.decision.engine_mock_impl import EngineMockImpl
from dev_double.providers.mock.decision.id_provider_mock_impl import IdProviderMockImpl


def test_clock_steps_and_advances():
    clock = ClockMockImpl(start=10.0, step=0.5)
    assert [clock.now(), clock.now()] == [10.0, 10.5]
    clock.advance(1.0)
    assert clock.now() == 12.0


def test_ids_count_up():
    ids = IdProviderMockImpl("x")
    assert [ids.new_id(), ids.new_id()] == ["x-1", "x-2"]


async def test_engine_scores_by_word_overlap():
    q = TLabelQuery(
        system="s",
        user="u v",
        labels=["refund", "rebooking"],
        input_text="I want my money refunded",
        question="?",
        descriptions=["refund money", "another flight seat"],
    )
    result = await EngineMockImpl().distribution(q)
    assert result.probabilities["refund"] > result.probabilities["rebooking"]
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert (result.usage.input_tokens, result.usage.output_tokens) == (3, 1)


async def test_engine_generates_a_json_record_from_labelled_lines():
    import json

    from dev_double.core.decision.t_generate import TGenerateQuery

    text = 'Invoice\nDue date: 2026-10-01\n{\n  "vendor": "Acme Corp",\n}'
    q = TGenerateQuery(
        system="s", user="u", max_tokens=64, json=True, input_text=text, fields=("due_date", "vendor", "total")
    )
    r = await EngineMockImpl().generate(q)
    assert json.loads(r.text) == {"due_date": "2026-10-01", "vendor": "Acme Corp", "total": None}
    assert (r.confidence, r.tokens, r.warnings) == (1.0, [], [])
    assert r.usage.output_tokens == 1
