"""EngineOpenAIImpl as an IGenerator: free-text values with confidence from logprobs."""

import json
import math

import httpx
import pytest
from _shared.resources import OLLAMA_MODEL, OLLAMA_URL, has_ollama_model
from _shared.timing import timed

from dev_double.core.decision.decider_basic_impl import DeciderBasicImpl
from dev_double.core.decision.decision_service_basic_impl import DecisionServiceBasicImpl
from dev_double.core.decision.errors import EngineError
from dev_double.core.decision.i_generator import IGenerator
from dev_double.core.decision.t_extract import TExtractField, TExtractRequest
from dev_double.core.decision.t_generate import TGenerateQuery
from dev_double.providers.mock.decision.clock_mock_impl import ClockMockImpl
from dev_double.providers.mock.decision.id_provider_mock_impl import IdProviderMockImpl
from dev_double.providers.openai.decision.engine_openai_impl import EngineOpenAIImpl

QUERY = TGenerateQuery(system="s", user="u", max_tokens=64)


def completion(content: str, logprobs: list[float] | None, finish: str = "stop") -> dict:
    choice: dict = {"message": {"role": "assistant", "content": content}, "finish_reason": finish}
    if logprobs is not None:
        choice["logprobs"] = {"content": [{"token": "t", "logprob": lp, "top_logprobs": []} for lp in logprobs]}
    return {"choices": [choice], "usage": {"prompt_tokens": 40, "completion_tokens": len(logprobs or [])}}


def engine_returning(payload: dict, status: int = 200) -> tuple[EngineOpenAIImpl, list]:
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return EngineOpenAIImpl("http://model/v1", "tiny", client=client), seen


async def test_generate_request_and_confidence_from_mean_logprob():
    engine, seen = engine_returning(completion("Acme Corp", [math.log(0.9), math.log(0.8)]))
    assert isinstance(engine, IGenerator)
    r = await engine.generate(QUERY)
    assert r.text == "Acme Corp"
    assert r.confidence == pytest.approx(math.sqrt(0.9 * 0.8))
    assert (r.usage.input_tokens, r.usage.output_tokens) == (40, 2) and r.warnings == []
    body = json.loads(seen[0].content)
    assert body["max_tokens"] == 64 and body["logprobs"] is True and body["temperature"] == 0
    assert body["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


async def test_generate_json_sends_response_format_and_returns_tokens():
    engine, seen = engine_returning(completion('{"a": 1}', [math.log(0.5), 0.0]))
    r = await engine.generate(TGenerateQuery(system="s", user="u", max_tokens=128, json=True))
    assert r.tokens == [("t", pytest.approx(math.log(0.5))), ("t", 0.0)]
    body = json.loads(seen[0].content)
    assert body["response_format"] == {"type": "json_object"} and body["max_tokens"] == 128
    engine, seen = engine_returning(completion("x", [0.0]))
    await engine.generate(QUERY)
    assert "response_format" not in json.loads(seen[0].content)


async def test_generate_without_logprobs_warns():
    engine, _ = engine_returning(completion("Acme", None))
    r = await engine.generate(QUERY)
    assert r.confidence == 1.0 and any("no logprobs" in w for w in r.warnings)


async def test_generate_truncated_warns():
    engine, _ = engine_returning(completion("a very long", [0.0], finish="length"))
    r = await engine.generate(QUERY)
    assert any("cut off" in w for w in r.warnings)


async def test_generate_http_error_raises():
    engine, _ = engine_returning({"error": "boom"}, status=500)
    with pytest.raises(EngineError, match="500"):
        await engine.generate(QUERY)
    engine, _ = engine_returning({"nothing": 1})
    with pytest.raises(EngineError, match="Unexpected"):
        await engine.generate(QUERY)


# ------------------------------------------------------------------ integration

INVOICE = """INVOICE #2291
From: Acme Corp, 12 Harbour Road, Dublin
Bill to: Globex Ltd

Consulting services, September ......... 1.000,00 EUR
VAT 20% ................................   200,00 EUR
Amount due: 1.200,00 EUR
Payment due by 1 October 2026. Status: unpaid."""


@pytest.mark.skipif(not has_ollama_model(), reason=f"no Ollama at {OLLAMA_URL} with {OLLAMA_MODEL}")
async def test_ollama_extracts_an_invoice():
    engine = EngineOpenAIImpl(f"{OLLAMA_URL}/v1", OLLAMA_MODEL)
    svc = DecisionServiceBasicImpl(DeciderBasicImpl(engine), ClockMockImpl(), IdProviderMockImpl())
    req = TExtractRequest(
        INVOICE,
        {
            "vendor": TExtractField("string", "Company that issued the invoice."),
            "total": TExtractField("number", "Amount due, without currency symbol."),
            "due_date": TExtractField("string", "Due date as YYYY-MM-DD."),
            "po_number": TExtractField("string", "Purchase order number."),
            "currency": TExtractField("enum", options=["EUR", "USD", "GBP"]),
            "paid": TExtractField("boolean", "Whether the invoice is already paid."),
        },
    )
    try:
        r = await timed(f"ollama {OLLAMA_MODEL} extract", lambda: svc.extract(req))
    finally:
        await svc.aclose()
    assert "Acme" in r.values["vendor"]
    assert r.values["total"] == pytest.approx(1200.0)
    assert r.values["due_date"] == "2026-10-01"
    assert r.values["po_number"] is None
    assert r.values["currency"] == "EUR" and r.values["paid"] is False
    assert 0 < r.fields["vendor"].confidence <= 1
