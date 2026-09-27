"""A model-free engine for CI and plumbing tests.

It scores each label by word overlap between the input and the label's
description. It is deterministic and instant, and its answers are only
loosely sensible. Use it to test your integration code, never its quality.

It also "generates" extraction values: for each field, the text after the
colon on the first input line whose label (before the colon) contains every
word of the field name, so ``due_date`` finds ``Due date: 2026-10-01``. In JSON
mode it returns one object with a key per field (null when not found) and
confidence 1.0; otherwise the first field's value, or NONE with confidence 0.5.
It returns no token logprobs.
"""

from __future__ import annotations

import json
import math
import re
from typing import Optional

from stunt_double.core.decision.i_engine import IEngine
from stunt_double.core.decision.i_generator import IGenerator
from stunt_double.core.decision.t_generate import TGenerateQuery, TGenerateResult
from stunt_double.core.decision.t_label_query import TLabelQuery, TLabelResult
from stunt_double.core.decision.t_usage import TUsage

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was "
    "were will with you your does do not no yes which what how".split()
)


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


class EngineMockImpl(IEngine, IGenerator):
    name = "mock"
    model = "mock-overlap"

    async def aclose(self) -> None:
        return None

    async def distribution(self, query: TLabelQuery) -> TLabelResult:
        input_words = _words(query.input_text)
        scores = []
        for description in query.descriptions:
            desc = _words(description)
            overlap = len(input_words & desc)
            scores.append(overlap / (1 + math.sqrt(len(desc))))
        top = max(scores)
        exps = [math.exp(4 * (s - top)) for s in scores]
        total = sum(exps)
        probs = {label: e / total for label, e in zip(query.labels, exps)}
        usage = TUsage(
            input_tokens=len(query.system.split()) + len(query.user.split()),
            output_tokens=1,
        )
        return TLabelResult(probs, usage)

    @staticmethod
    def _lookup(field: str, input_text: str) -> Optional[str]:
        wanted = set(_WORD.findall(field.lower()))
        for line in input_text.splitlines():
            key, colon, rest = line.partition(":")
            if colon and wanted and wanted <= set(_WORD.findall(key.lower())):
                return rest.strip().rstrip(",").strip().strip('"')
        return None

    async def generate(self, query: TGenerateQuery) -> TGenerateResult:
        found = {name: self._lookup(name, query.input_text) for name in query.fields}
        usage = TUsage(input_tokens=len(query.system.split()) + len(query.user.split()), output_tokens=1)
        if query.json:
            return TGenerateResult(json.dumps(found), 1.0, usage)
        first = next(iter(found.values()), None)
        return TGenerateResult(first, 1.0, usage) if first is not None else TGenerateResult("NONE", 0.5, usage)
