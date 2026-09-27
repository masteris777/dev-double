"""DeciderBasicImpl as an IRecordReader: all text fields in one JSON generation."""

import json
import math

from dev_double.core.decision import prompts
from dev_double.core.decision.decider_basic_impl import DeciderBasicImpl
from dev_double.core.decision.i_generator import IGenerator
from dev_double.core.decision.i_record_reader import IRecordReader
from dev_double.core.decision.t_extract import TExtractField, TFieldValue
from dev_double.core.decision.t_generate import TGenerateQuery, TGenerateResult
from dev_double.core.decision.t_usage import TUsage
from dev_double.core.decision.tracker import Tracker
from dev_double.providers.mock.decision.clock_mock_impl import ClockMockImpl
from dev_double.providers.mock.decision.engine_mock_impl import EngineMockImpl

FIELDS = {
    "vendor": TExtractField("string", "Company that issued the invoice."),
    "total": TExtractField("number"),
    "room_no": TExtractField("integer"),
}


class ScriptedGenerator:
    """Generates a fixed text and remembers the queries it saw."""

    name, model = "gen", "g1"

    def __init__(self, text: str, confidence: float = 0.9, tokens=None, warnings=None) -> None:
        self.result = TGenerateResult(text, confidence, TUsage(20, 9), warnings or [], tokens or [])
        self.queries: list[TGenerateQuery] = []

    async def distribution(self, query):  # pragma: no cover - not used
        raise AssertionError

    async def generate(self, query: TGenerateQuery) -> TGenerateResult:
        self.queries.append(query)
        return self.result

    async def aclose(self) -> None:
        return None


class LabelOnlyEngine:
    name, model = "labels", "l1"

    async def distribution(self, query):  # pragma: no cover - not used
        raise AssertionError

    async def aclose(self) -> None:
        return None


def test_record_prompt_is_the_tested_wording():
    q = prompts.extract_record("Invoice from Acme", FIELDS)
    assert q.system == (
        "You extract data from text into JSON. Output one JSON object with exactly the requested keys. "
        "Copy values from the text, converting them to the requested format. Use null only when the text "
        "gives no information for a key. Output only the JSON."
    )
    keys = (
        '- "vendor" (string): Company that issued the invoice.\n'
        '- "total" (number): total\n'
        '- "room_no" (integer): room no'
    )
    assert q.user == (
        f"Keys:\n{keys}\n\n<text>\nInvoice from Acme\n</text>\n\n"
        f"Fill in the keys from the text above:\n{keys}\n\nJSON:"
    )
    assert q.json is True and q.max_tokens == 48 + 40 * 3
    assert (q.input_text, q.fields) == ("Invoice from Acme", ("vendor", "total", "room_no"))
    many = {f"f{i}": TExtractField("string") for i in range(30)}
    assert prompts.extract_record("x", many).max_tokens == 512


async def test_reads_all_text_fields_in_one_generation():
    text = '{"vendor": "Acme Corp", "total": "1,200.50", "room_no": null}'
    tokens = [('{"vendor": ', 0.0), ('"Acme Corp"', math.log(0.8)), (', "total": "1,200.50", "room_no": null}', 0.0)]
    engine = ScriptedGenerator(text, confidence=0.95, tokens=tokens, warnings=["low"])
    d = DeciderBasicImpl(engine)
    assert isinstance(d, IRecordReader) and isinstance(engine, IGenerator)
    t = Tracker(ClockMockImpl())
    got = await d.read_record({"doc": "Acme"}, FIELDS, t)
    assert got == {
        "vendor": TFieldValue("Acme Corp", 0.8),
        "total": TFieldValue(1200.5, 1.0),
        "room_no": TFieldValue(None, 1.0),
    }
    assert len(engine.queries) == 1 and '"doc": "Acme"' in engine.queries[0].user
    assert t.warnings == ["low"] and (t.usage.input_tokens, t.usage.output_tokens) == (20, 9)


async def test_invalid_json_leaves_fields_empty():
    t = Tracker(ClockMockImpl())
    got = await DeciderBasicImpl(ScriptedGenerator("Sorry.")).read_record("x", FIELDS, t)
    assert got == {name: TFieldValue(None, 0.0) for name in FIELDS}
    assert len(t.warnings) == 1


async def test_engine_without_generation_leaves_fields_empty():
    t = Tracker(ClockMockImpl())
    got = await DeciderBasicImpl(LabelOnlyEngine()).read_record("x", {"vendor": TExtractField("string")}, t)
    assert got == {"vendor": TFieldValue(None, 0.0)}
    assert t.warnings == ["engine can't generate text: field vendor left empty"]


async def test_mock_engine_reads_labelled_lines():
    d = DeciderBasicImpl(EngineMockImpl())
    t = Tracker(ClockMockImpl())
    text = "INVOICE\nVendor: Acme Corp\nDue date: 2026-10-01\nTotal: EUR 1,200.50"
    fields = {
        "vendor": TExtractField("string"),
        "due_date": TExtractField("string"),
        "total": TExtractField("number"),
        "po_number": TExtractField("string"),
    }
    got = await d.read_record(text, fields, t)
    assert {k: v.value for k, v in got.items()} == {
        "vendor": "Acme Corp", "due_date": "2026-10-01", "total": 1200.5, "po_number": None,
    }
    assert json.loads(json.dumps({k: v.value for k, v in got.items()}))  # plain JSON values
