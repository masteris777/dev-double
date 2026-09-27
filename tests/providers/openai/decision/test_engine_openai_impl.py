import json
import math

import httpx
import pytest
from _shared.resources import OLLAMA_MODEL, OLLAMA_URL, has_ollama_model
from _shared.timing import timed

from stunt_double.core.decision import prompts
from stunt_double.core.decision.errors import EngineError
from stunt_double.core.decision.t_label_query import TLabelQuery
from stunt_double.core.decision.t_question import TBinaryQuestion, TChoiceQuestion
from stunt_double.providers.openai.decision.engine_openai_impl import EngineOpenAIImpl


def lp(p: float) -> float:
    return math.log(p)


def query(labels: list[str]) -> TLabelQuery:
    return TLabelQuery(
        system="s", user="u", labels=labels, input_text="x", question="q", descriptions=[""] * len(labels)
    )


def completion(positions: list[dict] | None, content: str = "") -> dict:
    choice: dict = {"message": {"role": "assistant", "content": content}}
    if positions is not None:
        choice["logprobs"] = {"content": positions}
    return {"choices": [choice], "usage": {"prompt_tokens": 50, "completion_tokens": 1}}


def engine_returning(payload: dict, status: int = 200) -> tuple[EngineOpenAIImpl, list]:
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return EngineOpenAIImpl("http://model/v1", "tiny", client=client), seen


async def test_reads_logprobs():
    positions = [
        {
            "token": "later",
            "logprob": lp(0.7),
            "top_logprobs": [
                {"token": "later", "logprob": lp(0.7)},
                {"token": "reply", "logprob": lp(0.2)},
                {"token": "archive", "logprob": lp(0.1)},
            ],
        }
    ]
    engine, seen = engine_returning(completion(positions, "later"))
    result = await engine.distribution(query(["reply_now", "later", "archive"]))
    assert result.probabilities == pytest.approx({"reply_now": 0.2, "later": 0.7, "archive": 0.1})
    assert result.usage.input_tokens == 50
    assert result.warnings == []
    sent = seen[0]
    assert sent.url.path == "/v1/chat/completions"
    body = json.loads(sent.content)
    assert body["logprobs"] is True and body["top_logprobs"] == 20 and body["temperature"] == 0
    assert body["model"] == "tiny" and engine.model == "tiny" and engine.name == "openai"


async def test_skips_leading_whitespace_token():
    positions = [
        {"token": "\n", "logprob": lp(0.9), "top_logprobs": [{"token": "\n", "logprob": lp(0.9)}]},
        {
            "token": "no",
            "logprob": lp(0.6),
            "top_logprobs": [{"token": "no", "logprob": lp(0.6)}, {"token": "yes", "logprob": lp(0.4)}],
        },
    ]
    engine, _ = engine_returning(completion(positions, "\nno"))
    result = await engine.distribution(query(["yes", "no"]))
    assert result.probabilities["no"] == pytest.approx(0.6)


async def test_without_logprobs_falls_back_with_warning():
    engine, _ = engine_returning(completion(None, "yes"))
    result = await engine.distribution(query(["yes", "no"]))
    assert result.probabilities == {"yes": 1.0, "no": 0.0}
    assert any("no logprobs" in w for w in result.warnings)


async def test_invalid_answer_is_uniform_with_warning():
    engine, _ = engine_returning(completion(None, "<think>"))
    result = await engine.distribution(query(["allow", "deny"]))
    assert result.probabilities == {"allow": 0.5, "deny": 0.5}
    assert any("valid label" in w for w in result.warnings)


async def test_http_error_raises():
    engine, _ = engine_returning({"error": "model not found"}, status=404)
    with pytest.raises(EngineError, match="404"):
        await engine.distribution(query(["yes", "no"]))


async def test_api_key_is_sent_as_bearer_token():
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=completion(None, "yes"))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    engine = EngineOpenAIImpl("http://model/v1", "tiny", api_key="k", client=client)
    await engine.distribution(query(["yes", "no"]))
    assert seen[0].headers["authorization"] == "Bearer k"


# ------------------------------------------------------------------ integration

ollama = pytest.mark.skipif(
    not has_ollama_model(), reason=f"no Ollama at {OLLAMA_URL} with {OLLAMA_MODEL}"
)


@ollama
async def test_ollama_answers_with_calibrated_labels():
    engine = EngineOpenAIImpl(f"{OLLAMA_URL}/v1", OLLAMA_MODEL)
    try:
        text = "The package arrived broken and I want my money back."
        yes = await timed(
            f"ollama {OLLAMA_MODEL} binary",
            lambda: engine.distribution(prompts.binary(text, TBinaryQuestion("Is the customer asking for a refund?"))),
        )
        q = TChoiceQuestion(
            "What does the customer want?",
            {"refund": "Money returned.", "rebooking": "A replacement flight.", "information": "Only information."},
        )
        pick = await timed(
            f"ollama {OLLAMA_MODEL} choice",
            lambda: engine.distribution(prompts.choice("My flight was cancelled, put me on the next one.", q)),
        )
    finally:
        await engine.aclose()
    assert yes.probabilities["yes"] > 0.5
    assert sum(pick.probabilities.values()) == pytest.approx(1.0)
    assert max(pick.probabilities, key=pick.probabilities.__getitem__) == "rebooking"
    assert yes.warnings == [] and yes.usage.input_tokens > 0
