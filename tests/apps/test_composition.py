import importlib.util

import pytest

from dev_double.apps.composition import build_decider, build_engine, build_service
from dev_double.apps.config import Settings
from dev_double.core.decision.decider_basic_impl import DeciderBasicImpl
from dev_double.core.decision.decision_service_basic_impl import DecisionServiceBasicImpl
from dev_double.providers.mock.decision.engine_mock_impl import EngineMockImpl
from dev_double.providers.openai.decision.engine_openai_impl import EngineOpenAIImpl
from dev_double.providers.systemone.decision.decider_system_one_impl import DeciderSystemOneImpl


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("DEV_DOUBLE_ENGINE", "systemone")
    monkeypatch.setenv("DEV_DOUBLE_SYSTEMONE_URL", "http://s1:9000")
    monkeypatch.setenv("DEV_DOUBLE_MAX_CONCURRENCY", "2")
    s = Settings.from_env()
    assert (s.engine, s.systemone_url, s.max_concurrency) == ("systemone", "http://s1:9000", 2)
    assert Settings().systemone_url == "http://localhost:8000"


async def test_builds_each_engine():
    assert isinstance(build_engine(Settings(engine="mock")), EngineMockImpl)
    openai = build_engine(Settings(engine="openai", model="m"))
    assert isinstance(openai, EngineOpenAIImpl) and openai.model == "m"
    await openai.aclose()

    d = build_decider(Settings(engine="mock"))
    assert isinstance(d, DeciderBasicImpl) and d.name == "mock"

    s1 = build_decider(Settings(engine="systemone", systemone_url="http://s1"))
    assert isinstance(s1, DeciderSystemOneImpl) and s1.name == "systemone"
    await s1.aclose()

    svc = build_service(Settings(engine="openai"), engine=EngineMockImpl())
    assert isinstance(svc, DecisionServiceBasicImpl) and svc.name == "mock"


def test_needle_needs_the_optional_extra():
    if importlib.util.find_spec("needle") is not None:
        pytest.skip("cactus-needle is installed")
    with pytest.raises(ImportError):
        build_decider(Settings(engine="needle"))


def test_unknown_engine():
    with pytest.raises(ValueError, match="openai, mock, systemone, needle"):
        build_decider(Settings(engine="fake"))
