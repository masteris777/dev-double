"""Which decision service answers a request, when requests may name their own model.

By default every request is answered by the one configured service and its
``model`` field is ignored. With ``honor_request_model`` the server builds one
service per requested model (openai: the chat model; systemone: the wire
``model``), keeps the most recently used few, and closes the rest. The mock and
needle engines have a single fixed model, so they ignore the field.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import AsyncIterator, Callable, Optional

from ...core.decision.i_decision_service import IDecisionService
from ..composition import build_service
from ..config import Settings

HONORING_ENGINES = ("openai", "systemone")  # engines whose model is chosen per request
MAX_CACHED = 8


@dataclass
class _Entry:
    service: IDecisionService
    users: int = 0
    evicted: bool = False


class ModelServices:
    def __init__(
        self,
        settings: Settings,
        default: IDecisionService,
        *,
        can_build: bool = True,
        factory: Optional[Callable[[Settings], IDecisionService]] = None,
        max_cached: int = MAX_CACHED,
    ) -> None:
        """``can_build`` is False when the default service was injected, so a service
        for another model can't be built from ``settings``."""
        self._settings = settings
        self._default = default
        self._can_build = can_build
        self._factory = factory
        self._max = max_cached
        self._cache: OrderedDict[str, _Entry] = OrderedDict()  # least recently used first

    def honors(self, model: Optional[str]) -> bool:
        """Whether a request naming ``model`` gets a service other than the default."""
        return (
            self._settings.honor_request_model
            and self._can_build
            and self._settings.engine in HONORING_ENGINES
            and bool(model and model.strip())
            and model != self._default.model
        )

    @asynccontextmanager
    async def use(self, model: Optional[str]) -> AsyncIterator[IDecisionService]:
        """The service to answer a request that names ``model`` (or none)."""
        if not self.honors(model):
            yield self._default
            return
        assert model is not None
        entry = await self._check_out(model)
        try:
            yield entry.service
        finally:
            entry.users -= 1
            if entry.evicted and entry.users == 0:
                await entry.service.aclose()

    async def _check_out(self, model: str) -> _Entry:
        entry = self._cache.pop(model, None)
        if entry is None:
            build = self._factory or build_service
            entry = _Entry(build(replace(self._settings, model=model)))
        self._cache[model] = entry
        entry.users += 1
        while len(self._cache) > self._max:
            _, old = self._cache.popitem(last=False)
            old.evicted = True
            if old.users == 0:  # otherwise its last user closes it
                await old.service.aclose()
        return entry

    async def aclose(self) -> None:
        entries, self._cache = list(self._cache.values()), OrderedDict()
        for entry in entries:
            await entry.service.aclose()
        await self._default.aclose()


def model_warning(requested: Optional[str], actual: str, honor: bool) -> Optional[str]:
    """What to tell the caller when the model that answered isn't the one they named."""
    if not requested or requested == actual:
        return None
    if honor:
        return f"requested model {requested!r} can't be selected on this engine; answered by {actual!r}."
    return (
        f"requested model {requested!r} was ignored; answered by the configured model {actual!r} "
        "(start the server with --honor-request-model to use the requested one)."
    )
