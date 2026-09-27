"""A real System 1 decision model (Kev, Laya) served at ``POST /v1/systemone``.

Each question is sent as-is (no dev-double prompt): these models answer
typed questions natively, so this implements ``IDecider`` directly.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx

from dev_double.core.decision.answer_shaping import shape_answer
from dev_double.core.decision.errors import EngineError
from dev_double.core.decision.i_decider import IDecider
from dev_double.core.decision.t_answer import TAnswer
from dev_double.core.decision.t_input import TInputValue
from dev_double.core.decision.t_question import TBinaryQuestion, TChoiceQuestion, TQuestion
from dev_double.core.decision.t_usage import TUsage
from dev_double.core.decision.tracker import Tracker

PATH = "/v1/systemone"


def to_wire(q: TQuestion) -> dict[str, Any]:
    """A core question in the System 1 wire format."""
    if isinstance(q, TBinaryQuestion):
        body: dict[str, Any] = {"type": "noul", "instructions": q.question}
        criteria = {k: v for k, v in (("true", q.yes), ("false", q.no)) if v}
        if criteria:
            body["criteria"] = criteria
        return body
    if isinstance(q, TChoiceQuestion):
        return {"type": "choice", "instructions": q.question, "criteria": q.options}
    return {"type": "score", "instructions": q.question, "criteria": q.levels}


def from_wire(q: TQuestion, answer: dict[str, Any]) -> dict[str, float]:
    """The wire answer as a distribution over the question's labels."""
    if isinstance(q, TBinaryQuestion):
        p = float(answer["noul"])
        return {"yes": p, "no": 1 - p}
    return {str(k): float(v) for k, v in answer["probabilities"].items()}


class DeciderSystemOneImpl(IDecider):
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: Optional[str] = None,
        *,
        model: Optional[str] = None,
        max_concurrency: int = 1,
        timeout: float = 120.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._name = name
        self._model = model or name
        self._base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._headers = headers
        self._limit = asyncio.Semaphore(max_concurrency)

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    async def ask(
        self, value: TInputValue, question: TQuestion, tracker: Tracker, label: str = ""
    ) -> TAnswer:
        payload = {"state": value, "questions": {"q": to_wire(question)}}
        url = self._base_url + PATH
        try:
            async with self._limit:
                r = await self._client.post(url, json=payload, headers=self._headers)
        except httpx.HTTPError as exc:
            raise EngineError(f"Could not reach System 1 server at {url}: {exc}") from exc
        if r.status_code >= 400:
            raise EngineError(f"System 1 server returned {r.status_code}: {r.text[:500]}")
        data = r.json()
        try:
            probs = from_wire(question, data["answers"]["q"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EngineError(f"Unexpected response from System 1 server: {str(data)[:500]}") from exc
        usage = data.get("usage") or {}
        tracker.usage.add(
            TUsage(input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0))
        )
        # The server's probabilities are passed through unrounded, as it sent them.
        return shape_answer(question, probs, round_probs=False)

    async def aclose(self) -> None:
        await self._client.aclose()
