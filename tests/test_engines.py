import math

import httpx
import pytest

from stunt_double.engines import EngineError, LabelQuery, OpenAICompatEngine
from stunt_double.engines.base import Position, label_distribution


def lp(p: float) -> float:
    return math.log(p)


def query(labels: list[str]) -> LabelQuery:
    return LabelQuery(
        system="s", user="u", labels=labels, input_text="x", question="q", descriptions=[""] * len(labels)
    )


def pos(token: str, *candidates: tuple[str, float]) -> Position:
    return Position(token, [(t, lp(p)) for t, p in candidates])


def test_merges_case_and_whitespace_variants():
    positions = [pos("yes", ("yes", 0.5), (" Yes", 0.2), ("No", 0.1), ("Maybe", 0.2))]
    dist, mass = label_distribution(positions, ["yes", "no"])
    assert mass == pytest.approx(0.8)
    assert dist["yes"] == pytest.approx(0.875)
    assert dist["no"] == pytest.approx(0.125)


def test_matches_option_names_ignoring_punctuation_and_separators():
    positions = [pos("Reply", ("Reply", 0.7), ("archive.", 0.2), ("later", 0.1))]
    dist, _ = label_distribution(positions, ["reply_now", "later", "archive"])
    assert dist == pytest.approx({"reply_now": 0.7, "later": 0.1, "archive": 0.2})


def test_shared_first_token_is_split_by_next_position():
    # "re" fits both refund and rebooking; the next token decides.
    positions = [
        pos("re", ("re", 0.9), ("information", 0.1)),
        pos("booking", ("booking", 0.75), ("fund", 0.25)),
    ]
    dist, mass = label_distribution(positions, ["refund", "rebooking", "information"])
    assert mass == pytest.approx(1.0)
    assert dist == pytest.approx({"refund": 0.225, "rebooking": 0.675, "information": 0.1})


def test_label_that_is_a_prefix_of_another():
    positions = [
        pos("info", ("info", 1.0)),
        pos("<|im_end|>", ("<|im_end|>", 0.6), ("rmation", 0.4)),
    ]
    dist, _ = label_distribution(positions, ["info", "information"])
    assert dist == pytest.approx({"info": 0.6, "information": 0.4})


def test_leading_whitespace_token_is_skipped():
    positions = [pos("\n", ("\n", 0.9)), pos("no", ("no", 0.6), ("yes", 0.4))]
    dist, _ = label_distribution(positions, ["yes", "no"])
    assert dist["no"] == pytest.approx(0.6)


def test_off_script_answer_has_zero_mass():
    _, mass = label_distribution([pos("The", ("The", 0.9), ("I", 0.1))], ["yes", "no"])
    assert mass == 0


def completion(positions: list[dict] | None, content: str = "") -> dict:
    choice: dict = {"message": {"role": "assistant", "content": content}}
    if positions is not None:
        choice["logprobs"] = {"content": positions}
    return {"choices": [choice], "usage": {"prompt_tokens": 50, "completion_tokens": 1}}


def engine_returning(payload: dict, status: int = 200) -> tuple[OpenAICompatEngine, list]:
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatEngine("http://model/v1", "tiny", client=client), seen


async def test_openai_engine_reads_logprobs():
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
    body = __import__("json").loads(sent.content)
    assert body["logprobs"] is True and body["top_logprobs"] == 20 and body["temperature"] == 0


async def test_openai_engine_skips_leading_whitespace_token():
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


async def test_openai_engine_without_logprobs_falls_back_with_warning():
    engine, _ = engine_returning(completion(None, "yes"))
    result = await engine.distribution(query(["yes", "no"]))
    assert result.probabilities == {"yes": 1.0, "no": 0.0}
    assert any("no logprobs" in w for w in result.warnings)


async def test_openai_engine_invalid_answer_is_uniform_with_warning():
    engine, _ = engine_returning(completion(None, "<think>"))
    result = await engine.distribution(query(["allow", "deny"]))
    assert result.probabilities == {"allow": 0.5, "deny": 0.5}
    assert any("valid label" in w for w in result.warnings)


async def test_openai_engine_http_error_raises():
    engine, _ = engine_returning({"error": "model not found"}, status=404)
    with pytest.raises(EngineError, match="404"):
        await engine.distribution(query(["yes", "no"]))
