"""A real System 1 decision model (Kev, Laya, Ollama's System One models) served
at ``POST /v1/systemone``.

Each question is sent as-is (no dev-double prompt): these models answer
typed questions natively, so this implements ``IDecider`` directly. Questions
about the same input travel together, one call per ``MAX_QUESTIONS``. Images are
forwarded as-is; the server says whether its model can read them.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx

from dev_double.core.decision import systemone_wire as wire
from dev_double.core.decision.answer_shaping import shape_answer
from dev_double.core.decision.errors import EngineError
from dev_double.core.decision.i_decider import IDecider
from dev_double.core.decision.t_answer import TAnswer
from dev_double.core.decision.t_input import TInputValue
from dev_double.core.decision.t_question import TQuestion
from dev_double.core.decision.tracker import Tracker

PATH = "/v1/systemone"


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

    @property
    def supports_images(self) -> bool:
        return True  # whether the model can read them is for the server to say

    async def ask(
        self,
        value: TInputValue,
        question: TQuestion,
        tracker: Tracker,
        label: str = "",
        images: tuple[str, ...] = (),
    ) -> TAnswer:
        answers = await self.ask_many(value, {"q": question}, tracker, label, images)
        return answers["q"]

    async def ask_many(
        self,
        value: TInputValue,
        questions: dict[str, TQuestion],
        tracker: Tracker,
        label: str = "",
        images: tuple[str, ...] = (),
    ) -> dict[str, TAnswer]:
        names = list(questions)
        chunks = [names[i : i + wire.MAX_QUESTIONS] for i in range(0, len(names), wire.MAX_QUESTIONS)]
        parts = await asyncio.gather(
            *(self._ask_chunk(value, {n: questions[n] for n in chunk}, tracker, images) for chunk in chunks)
        )
        answers: dict[str, TAnswer] = {}
        for part in parts:
            answers.update(part)
        return {name: answers[name] for name in names}

    async def _ask_chunk(
        self, value: TInputValue, questions: dict[str, TQuestion], tracker: Tracker, images: tuple[str, ...]
    ) -> dict[str, TAnswer]:
        """One HTTP call carrying all of ``questions`` (and the ``images`` they share)."""
        payload = wire.request_to_wire(self._model, value, questions, images)
        url = self._base_url + PATH
        try:
            async with self._limit:
                r = await self._client.post(url, json=payload, headers=self._headers)
        except httpx.HTTPError as exc:
            raise EngineError(f"Could not reach System 1 server at {url}: {exc}") from exc
        if r.status_code >= 400:
            raise EngineError(f"System 1 server returned {r.status_code}: {r.text[:500]}")
        try:
            data = r.json()
            wire_answers = data["answers"]
            usage = wire.usage_from_wire(data.get("usage"))
            answers: dict[str, TAnswer] = {}
            for name, question in questions.items():
                if name not in wire_answers:
                    raise EngineError(f"System 1 server returned no answer for question {name!r}.")
                probs = wire.probabilities_from_wire(question, wire_answers[name], f"answers.{name}")
                # The server's probabilities are passed through unrounded, as it sent them.
                answers[name] = shape_answer(question, probs, round_probs=False)
        except (KeyError, TypeError, ValueError) as exc:
            raise EngineError(f"Unexpected response from System 1 server: {r.text[:500]}") from exc
        tracker.usage.add(usage)  # once per call, however many questions it carried
        return answers

    async def aclose(self) -> None:
        await self._client.aclose()
