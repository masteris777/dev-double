"""Engine for any server with an OpenAI-style /chat/completions endpoint that
returns logprobs: Ollama (>= 0.12.11), vLLM, llama.cpp server, LM Studio, or a
hosted provider your company already approved.

The model is asked to reply with a single label, and the answer is read from
the probabilities the model gave each candidate token. One short
completion per question; no text generation, no JSON parsing.
"""

from __future__ import annotations

from typing import Optional

import httpx

from stunt_double.core.decision.errors import EngineError
from stunt_double.core.decision.i_engine import IEngine
from stunt_double.core.decision.label_scoring import label_distribution, label_from_text, one_hot, uniform
from stunt_double.core.decision.t_label_query import TLabelQuery, TLabelResult, TPosition
from stunt_double.core.decision.t_usage import TUsage

# Enough tokens to get past a leading newline and to tell apart options that
# share their first token ("re" + "fund" vs "re" + "booking").
MAX_TOKENS = 8
TOP_LOGPROBS = 20
LOW_MASS = 0.5


class EngineOpenAIImpl(IEngine):
    name = "openai"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await self._client.aclose()

    async def distribution(self, query: TLabelQuery) -> TLabelResult:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": query.system},
                {"role": "user", "content": query.user},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
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
        content = (choice.get("message") or {}).get("content") or ""
        positions = ((choice.get("logprobs") or {}).get("content")) or []
        return self._read(positions, content, query.labels, usage)

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
