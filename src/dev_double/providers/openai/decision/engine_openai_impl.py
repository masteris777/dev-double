"""Engine for any server with an OpenAI-style /chat/completions endpoint that
returns logprobs: Ollama (>= 0.12.11), vLLM, llama.cpp server, LM Studio, or a
hosted provider your company already approved.

The model is asked to reply with a single label, and the answer is read from
the probabilities the model gave each candidate token. One short
completion per question; no text generation, no JSON parsing.

It is also an ``IGenerator``: for extraction fields that need free text
(names, amounts, dates) it generates one JSON object, and returns the tokens'
probabilities so each value gets its own confidence.

Questions about images go to a separate ``vision_model`` (a text model can't read
them): the images are sent first in the user message, so questions about the same
images share a prompt prefix that the server can reuse instead of re-reading them.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import httpx

from dev_double.core.decision.errors import EngineError, UnsupportedRequestError
from dev_double.core.decision.i_engine import IEngine
from dev_double.core.decision.i_generator import IGenerator
from dev_double.core.decision.images import image_mime
from dev_double.core.decision.label_scoring import label_distribution, label_from_text, one_hot, uniform
from dev_double.core.decision.t_generate import TGenerateQuery, TGenerateResult
from dev_double.core.decision.t_label_query import TLabelQuery, TLabelResult, TPosition
from dev_double.core.decision.t_usage import TUsage

# Enough tokens to get past a leading newline and to tell apart options that
# share their first token ("re" + "fund" vs "re" + "booking").
MAX_TOKENS = 8
# The most alternatives per token that OpenAI-compatible servers return. Each
# label's probability comes from the alternatives at the answer's first token,
# so a question with more labels than this would silently score the unseen ones 0.
TOP_LOGPROBS = 20
LOW_MASS = 0.5
NO_VISION_MODEL = (
    "this engine needs a vision model to read images: set --vision-model "
    "(or DEV_DOUBLE_VISION_MODEL), for example qwen2.5vl:7b"
)


class EngineOpenAIImpl(IEngine, IGenerator):
    name = "openai"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        client: Optional[httpx.AsyncClient] = None,
        vision_model: Optional[str] = None,
    ) -> None:
        self._model = model
        self._vision_model = vision_model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @property
    def model(self) -> str:
        return self._model

    @property
    def supports_images(self) -> bool:
        return self._vision_model is not None

    def model_for(self, images: bool) -> str:
        return self._vision_model if images and self._vision_model else self._model

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _user_content(user: str, images: tuple[str, ...]) -> Union[str, list[dict]]:
        """The user message: the text, or with images, content parts with the images first."""
        if not images:
            return user
        parts: list[dict] = []
        for image in images:
            mime = image_mime(image)
            if mime is None:
                raise UnsupportedRequestError("images must be base64 PNG, JPEG, or WebP")
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}"}})
        return [*parts, {"type": "text", "text": user}]

    async def _complete(
        self, system: str, user: str, images: tuple[str, ...] = (), **options: object
    ) -> tuple[dict, TUsage]:
        """One /chat/completions call; returns the first choice and the usage."""
        if images and not self._vision_model:
            raise UnsupportedRequestError(NO_VISION_MODEL)
        body = {
            "model": self.model_for(bool(images)),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": self._user_content(user, images)},
            ],
            "temperature": 0,
            "logprobs": True,
            **options,
        }
        try:
            response = await self._client.post(self._url, json=body, headers=self._headers)
        except httpx.HTTPError as exc:
            raise EngineError(f"Could not reach model server at {self._url}: {exc}") from exc
        if response.status_code >= 400:
            raise EngineError(
                f"Model server returned {response.status_code}: {response.text[:500]}"
            )

        data = response.json()
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError) as exc:
            raise EngineError(f"Unexpected response from model server: {str(data)[:500]}") from exc

        raw_usage = data.get("usage") or {}
        usage = TUsage(
            input_tokens=raw_usage.get("prompt_tokens", 0),
            output_tokens=raw_usage.get("completion_tokens", 0),
        )
        return choice, usage

    @staticmethod
    def _check_scorable(labels: list[str]) -> None:
        """Refuses label sets whose probabilities this engine would read wrongly."""
        if len(labels) > TOP_LOGPROBS:
            raise UnsupportedRequestError(
                f"this engine supports at most {TOP_LOGPROBS} options; got {len(labels)}"
            )
        # Numbers such as "1" and "10": when the model says "1" the server doesn't
        # show whether it stopped there, so the two can't be told apart.
        if all(label.isdigit() for label in labels):
            for a in labels:
                for b in labels:
                    if a != b and b.startswith(a):
                        raise UnsupportedRequestError(
                            f"this engine can't tell the numeric options {a!r} and {b!r} apart; "
                            "use at most 10 levels, or non-numeric option keys"
                        )

    async def distribution(self, query: TLabelQuery) -> TLabelResult:
        self._check_scorable(query.labels)
        choice, usage = await self._complete(
            query.system, query.user, query.images, max_tokens=MAX_TOKENS, top_logprobs=TOP_LOGPROBS
        )
        content = (choice.get("message") or {}).get("content") or ""
        positions = ((choice.get("logprobs") or {}).get("content")) or []
        return self._read(positions, content, query.labels, usage)

    async def generate(self, query: TGenerateQuery) -> TGenerateResult:
        """Free text (or one JSON object) at temperature 0; confidence is exp(mean
        logprob) of the generated tokens, which are returned for per-value scoring."""
        options: dict = {"max_tokens": query.max_tokens, "top_logprobs": 1}
        if query.json:
            options["response_format"] = {"type": "json_object"}
        choice, usage = await self._complete(query.system, query.user, **options)
        content = (choice.get("message") or {}).get("content") or ""
        positions = ((choice.get("logprobs") or {}).get("content")) or []
        tokens = [(p.get("token", ""), float(p["logprob"])) for p in positions if "logprob" in p]
        warnings: list[str] = []
        if tokens:
            confidence = min(1.0, math.exp(sum(lp for _, lp in tokens) / len(tokens)))
        else:
            confidence = 1.0
            warnings.append(
                "The model server returned no logprobs, so extracted values' confidence is a "
                "placeholder 1.0. Use a server with logprobs support (e.g. Ollama >= 0.12.11)."
            )
        if choice.get("finish_reason") == "length":
            warnings.append(f"The output was cut off at {query.max_tokens} tokens.")
        return TGenerateResult(content, confidence, usage, warnings, tokens)

    @staticmethod
    def _read(raw_positions: list[dict], content: str, labels: list[str], usage: TUsage) -> TLabelResult:
        warnings: list[str] = []
        positions = [
            TPosition(
                token=pos.get("token", ""),
                candidates=[(c["token"], c["logprob"]) for c in pos.get("top_logprobs") or []],
            )
            for pos in raw_positions
        ]
        dist, mass = label_distribution(positions, labels)
        if mass > 0:
            if mass < LOW_MASS:
                warnings.append(
                    f"Only {mass:.0%} of the model's probability landed on a valid label; "
                    "this answer is unreliable. Try a larger or instruction-tuned model."
                )
            return TLabelResult(dist, usage, warnings)

        # No usable logprobs: fall back to the generated text.
        if not positions:
            warnings.append(
                "The model server returned no logprobs, so probabilities are 0/1 guesses, "
                "not calibrated. Use a server with logprobs support (e.g. Ollama >= 0.12.11)."
            )
        match = label_from_text(content, labels)
        if match is not None:
            return TLabelResult(one_hot(labels, match), usage, warnings)
        warnings.append(
            f"The model did not answer with a valid label (got {content[:40]!r}); "
            "returning a uniform distribution. Reasoning models that emit <think> "
            "first are not supported; use a non-thinking instruct model."
        )
        return TLabelResult(uniform(labels), usage, warnings)
