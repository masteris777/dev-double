import json
import math
from dataclasses import replace

import httpx
import pytest
from _shared import images
from _shared.resources import OLLAMA_MODEL, OLLAMA_URL, VISION_MODEL, has_ollama_model
from _shared.timing import timed

from dev_double.core.decision import prompts
from dev_double.core.decision.errors import EngineError, UnsupportedRequestError
from dev_double.core.decision.t_label_query import TLabelQuery
from dev_double.core.decision.t_question import TBinaryQuestion, TChoiceQuestion
from dev_double.providers.openai.decision.engine_openai_impl import EngineOpenAIImpl


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


async def test_more_than_20_options_is_a_client_error_not_a_wrong_answer():
    engine, seen = engine_returning(completion(None, "o0"))
    labels = [f"option{i}" for i in range(26)]
    with pytest.raises(UnsupportedRequestError, match="at most 20 options; got 26"):
        await engine.distribution(query(labels))
    assert seen == []  # refused before calling the model
    await engine.distribution(query(labels[:20]))  # 20 is the limit
    assert len(seen) == 1


async def test_numeric_options_where_one_starts_another_are_refused():
    engine, seen = engine_returning(completion(None, "1"))
    with pytest.raises(UnsupportedRequestError, match="'1' and '10'"):
        await engine.distribution(query([str(i) for i in range(12)]))
    assert seen == []
    await engine.distribution(query([str(i) for i in range(10)]))  # single digits are fine
    assert len(seen) == 1


# ------------------------------------------------------------------ integration

ollama = pytest.mark.skipif(
    not has_ollama_model(), reason=f"no Ollama at {OLLAMA_URL} with {OLLAMA_MODEL}"
)
vision = pytest.mark.skipif(
    not has_ollama_model(VISION_MODEL), reason=f"no Ollama at {OLLAMA_URL} with {VISION_MODEL}"
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


# ------------------------------------------------------------------ images


def image_query(*imgs: str) -> TLabelQuery:
    return TLabelQuery(
        system="s", user="the question", labels=["red", "blue"], input_text="x", question="q",
        descriptions=["", ""], images=imgs,
    )


RED_ANSWER = completion(
    [{"token": "red", "logprob": lp(0.9), "top_logprobs": [{"token": "red", "logprob": lp(0.9)}, {"token": "blue", "logprob": lp(0.1)}]}],
    "red",
)


def vision_engine(payload: dict = RED_ANSWER, vision_model: str | None = "seer") -> tuple[EngineOpenAIImpl, list]:
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return EngineOpenAIImpl("http://model/v1", "tiny", client=client, vision_model=vision_model), seen


async def test_images_go_first_in_the_user_content_to_the_vision_model():
    engine, seen = vision_engine()
    result = await engine.distribution(image_query(images.PNG, images.JPEG, images.WEBP))
    assert result.probabilities == pytest.approx({"red": 0.9, "blue": 0.1})
    body = seen[0]
    assert body["model"] == "seer"
    assert body["messages"][0] == {"role": "system", "content": "s"}
    assert body["messages"][1] == {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{images.PNG}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images.JPEG}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{images.WEBP}"}},
            {"type": "text", "text": "the question"},
        ],
    }
    assert body["logprobs"] is True and body["top_logprobs"] == 20


async def test_questions_about_the_same_image_share_a_prompt_prefix():
    engine, seen = vision_engine()
    await engine.distribution(image_query(images.PNG))
    other = image_query(images.PNG)
    other.user = "a different question"
    await engine.distribution(other)
    first, second = ([m["messages"][0], m["messages"][1]["content"][0]] for m in seen)
    assert first == second


async def test_without_images_the_text_model_gets_a_plain_string():
    engine, seen = vision_engine()
    await engine.distribution(image_query())
    assert seen[0]["model"] == "tiny"
    assert seen[0]["messages"][1] == {"role": "user", "content": "the question"}


async def test_images_without_a_vision_model_fail_clearly_and_call_nothing():
    engine, seen = vision_engine(vision_model=None)
    with pytest.raises(UnsupportedRequestError, match="--vision-model"):
        await engine.distribution(image_query(images.PNG))
    assert seen == []


async def test_an_unrecognised_image_is_refused_before_the_call():
    engine, seen = vision_engine()
    with pytest.raises(UnsupportedRequestError, match="PNG, JPEG, or WebP"):
        await engine.distribution(image_query("aGVsbG8="))
    assert seen == []


def test_vision_support_and_the_model_that_answers():
    plain, _ = vision_engine(vision_model=None)
    seer, _ = vision_engine()
    assert (plain.supports_images, seer.supports_images) == (False, True)
    assert [plain.model_for(False), plain.model_for(True)] == ["tiny", "tiny"]
    assert [seer.model_for(False), seer.model_for(True)] == ["tiny", "seer"]
    assert seer.model == "tiny"


@vision
async def test_ollama_reads_a_solid_red_image():
    engine = EngineOpenAIImpl(f"{OLLAMA_URL}/v1", OLLAMA_MODEL, vision_model=VISION_MODEL)
    q = TChoiceQuestion(
        "What colour is the image?",
        {"red": "The image is red.", "blue": "The image is blue.", "green": "The image is green."},
    )
    try:
        result = await timed(
            f"ollama {VISION_MODEL} image choice",
            lambda: engine.distribution(replace(prompts.choice("(see the image)", q), images=(images.RED_PNG,))),
        )
    finally:
        await engine.aclose()
    assert max(result.probabilities, key=result.probabilities.__getitem__) == "red"
    assert result.usage.input_tokens > 0
