"""Cactus Needle, a tiny on-device tool-calling model, as a decider.

Needle isn't a decision model: it fills tool calls. Its docs treat
classification as extraction with an enum field, so each question becomes one
tool whose `label` argument may only be one of the allowed labels. Needle
returns one calibrated confidence for the call, not a distribution, so the
other labels share the remainder evenly. Rerank uses Needle's embeddings.

Needs the optional extra: `pip install "stunt-double[needle]"`. ``needle`` is
imported lazily, so the package imports without it. Telemetry is off by
default (NEEDLE_TELEMETRY=0, DO_NOT_TRACK=1); the library reads
NEEDLE3_LIB_PATH itself if you need to point it at a local build.
"""

from __future__ import annotations

import math
import os
from typing import Any, Literal, Optional

from stunt_double.core.decision import prompts
from stunt_double.core.decision.answer_shaping import question_labels, shape_answer
from stunt_double.core.decision.confidence import DIGITS
from stunt_double.core.decision.i_decider import IDecider
from stunt_double.core.decision.i_reranker import IReranker
from stunt_double.core.decision.t_answer import TAnswer
from stunt_double.core.decision.t_input import TInputValue
from stunt_double.core.decision.t_question import TQuestion
from stunt_double.core.decision.tracker import Tracker


def _import_needle() -> Any:
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    import needle

    return needle


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1)


class DeciderNeedleImpl(IDecider, IReranker):
    name = "needle"
    model = "needle3"

    def __init__(self) -> None:
        self._needle = _import_needle()
        self._agents: dict[tuple, Any] = {}
        self._embedder: Optional[Any] = None

    def _agent(self, q: TQuestion, labels: dict[str, str]) -> Any:
        from pydantic import Field, create_model  # the tool schema Needle expects

        key = (q.question, tuple(labels.items()))
        if key not in self._agents:
            options = "; ".join(f"{k}: {v}" for k, v in labels.items())
            tool = create_model(
                "answer",
                __doc__=f"Record the answer to: {q.question}",
                label=(Literal[tuple(labels)], Field(description=f"Exactly one of: {options}")),
            )
            system = f"{q.question}\nAllowed labels: {options}\nAlways call answer with one label."
            self._agents[key] = self._needle.Needle(tools=[tool], system=system, stateless=True)
        return self._agents[key]

    async def ask(
        self, value: TInputValue, question: TQuestion, tracker: Tracker, label: str = ""
    ) -> TAnswer:
        labels = question_labels(question)
        r = self._agent(question, labels).complete(prompts.render_input(value))
        calls = r.get("function_calls") or r.get("suppressed_calls") or []
        chosen = (calls[0].get("arguments") or {}).get("label") if calls else None
        conf = r.get("confidence")
        conf = 1.0 if conf is None else float(conf)
        if chosen in labels:
            rest = (1 - conf) / (len(labels) - 1)
            probs = {k: (conf if k == chosen else rest) for k in labels}
        else:
            tracker.warn("needle made no valid call", label)
            probs = {k: 1 / len(labels) for k in labels}
        return shape_answer(question, probs)

    async def rerank_scores(self, query: str, documents: list[str], tracker: Tracker) -> list[float]:
        if self._embedder is None:
            self._embedder = self._needle.Needle(stateless=True)
        embed = self._embedder.embed
        q = embed(query)
        return [round(_cosine(q, embed(text)), DIGITS) for text in documents]

    async def aclose(self) -> None:
        for agent in [*self._agents.values(), *([self._embedder] if self._embedder else [])]:
            agent.close()
