"""The composition root: the only place that picks implementations.

To add an engine: implement ``IEngine`` (a model that scores labels) or
``IDecider`` (a model that answers typed questions natively) under
``providers/<name>/decision/``, then add a branch here and to ``ENGINES``.
"""

from __future__ import annotations

from typing import Optional

from ..core.decision.decider_basic_impl import DeciderBasicImpl
from ..core.decision.decision_service_basic_impl import DecisionServiceBasicImpl
from ..core.decision.i_decider import IDecider
from ..core.decision.i_engine import IEngine
from ..providers.std.decision.clock_std_impl import ClockStdImpl
from ..providers.std.decision.id_provider_std_impl import IdProviderStdImpl
from .config import ENGINES, Settings


def _unknown(engine: str) -> ValueError:
    return ValueError(f"Unknown engine {engine!r}; expected one of: {', '.join(ENGINES)}.")


def build_engine(settings: Settings) -> IEngine:
    """The label-scoring engine for prompt-based engines (openai, mock)."""
    if settings.engine == "mock":
        from ..providers.mock.decision.engine_mock_impl import EngineMockImpl

        return EngineMockImpl()
    if settings.engine == "openai":
        from ..providers.openai.decision.engine_openai_impl import EngineOpenAIImpl

        return EngineOpenAIImpl(
            base_url=settings.base_url,
            model=settings.model,
            api_key=settings.api_key,
            timeout=settings.timeout,
        )
    raise _unknown(settings.engine)


def build_decider(settings: Settings, engine: Optional[IEngine] = None) -> IDecider:
    if engine is not None or settings.engine in ("openai", "mock"):
        return DeciderBasicImpl(engine or build_engine(settings), settings.max_concurrency)
    if settings.engine == "systemone":
        from ..providers.systemone.decision.decider_system_one_impl import DeciderSystemOneImpl

        return DeciderSystemOneImpl(
            "systemone",
            settings.systemone_url,
            settings.api_key,
            max_concurrency=settings.max_concurrency,
            timeout=settings.timeout,
        )
    if settings.engine == "needle":
        from ..providers.needle.decision.decider_needle_impl import DeciderNeedleImpl

        return DeciderNeedleImpl()
    raise _unknown(settings.engine)


def build_service(
    settings: Settings,
    engine: Optional[IEngine] = None,
    decider: Optional[IDecider] = None,
) -> DecisionServiceBasicImpl:
    """The use-case service. Pass ``engine`` or ``decider`` to override the configured one."""
    return DecisionServiceBasicImpl(
        decider or build_decider(settings, engine), ClockStdImpl(), IdProviderStdImpl()
    )
