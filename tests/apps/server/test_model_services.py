"""ModelServices: one decision service per requested model, a small LRU, evicted ones closed."""

import asyncio

from dev_double.apps.config import Settings
from dev_double.apps.server.model_services import ModelServices, model_warning


class FakeService:
    def __init__(self, model: str) -> None:
        self.name, self.model = "fake", model
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def services(max_cached: int = 2, engine: str = "openai", honor: bool = True, can_build: bool = True):
    built: list[FakeService] = []

    def factory(settings: Settings) -> FakeService:
        built.append(FakeService(settings.model))
        return built[-1]

    default = FakeService("configured")
    pool = ModelServices(
        Settings(engine=engine, honor_request_model=honor),
        default,
        can_build=can_build,
        factory=factory,
        max_cached=max_cached,
    )
    return pool, default, built


async def test_a_service_is_built_once_per_model_and_reused():
    pool, default, built = services()
    async with pool.use("a") as first:
        pass
    async with pool.use("a") as again:
        pass
    assert first is again and first.model == "a" and len(built) == 1


async def test_the_least_recently_used_service_is_closed_on_eviction():
    pool, _, built = services(max_cached=2)
    for model in ("a", "b"):
        async with pool.use(model):
            pass
    async with pool.use("a"):  # a is now the most recent
        pass
    async with pool.use("c"):  # evicts b
        pass
    by_model = {s.model: s for s in built}
    assert by_model["b"].closed and not by_model["a"].closed and not by_model["c"].closed


async def test_a_service_in_use_is_closed_only_after_its_last_request_finishes():
    pool, _, built = services(max_cached=1)
    async with pool.use("a") as a:
        async with pool.use("b"):  # evicts a while a request is still using it
            assert not a.closed
        assert not a.closed
    assert a.closed and not built[1].closed


async def test_concurrent_requests_for_a_new_model_share_one_service():
    pool, _, built = services()

    async def one() -> FakeService:
        async with pool.use("a") as svc:
            await asyncio.sleep(0)
            return svc

    first, second = await asyncio.gather(one(), one())
    assert first is second and len(built) == 1


async def test_the_default_service_answers_when_the_model_is_not_honored():
    for kwargs, model in (
        ({"honor": False}, "a"),
        ({"engine": "mock"}, "a"),
        ({"engine": "needle"}, "a"),
        ({"can_build": False}, "a"),
        ({}, None),
        ({}, ""),
        ({}, "configured"),
    ):
        pool, default, built = services(**kwargs)
        async with pool.use(model) as svc:
            assert svc is default
        assert built == []


async def test_aclose_closes_everything():
    pool, default, built = services()
    async with pool.use("a"):
        pass
    await pool.aclose()
    assert default.closed and built[0].closed


def test_model_warning_wording():
    assert model_warning(None, "x", honor=False) is None
    assert model_warning("x", "x", honor=True) is None
    assert "--honor-request-model" in model_warning("a", "b", honor=False)
    assert "can't be selected" in model_warning("a", "b", honor=True)
