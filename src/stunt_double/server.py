"""The HTTP server."""

from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from . import __version__, usecases
from .config import Settings
from .core import Decider
from .engines import Engine, EngineError
from .schemas import (
    ClassifyRequest,
    ClassifyResponse,
    DecideRequest,
    DecideResponse,
    GateRequest,
    GateResponse,
    GuardRequest,
    GuardResponse,
    JudgeRequest,
    JudgeResponse,
    RerankRequest,
    RerankResponse,
    RouteRequest,
    RouteResponse,
)


class Recorder:
    """Appends every API call to a JSONL file, so you can replay the same
    traffic against the real engine later and compare."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()

    def write(self, entry: dict) -> None:
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock, open(self._path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        eng = engine or settings.build_engine()
        app.state.decider = Decider(eng, max_concurrency=settings.max_concurrency)
        try:
            yield
        finally:
            await eng.aclose()

    app = FastAPI(
        title="stunt-double",
        version=__version__,
        description=(
            "A local stand-in for fast decision-model and reranking APIs. "
            "Build against it now; point your client at the real engine later."
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(EngineError)
    async def _engine_error(_: Request, exc: EngineError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"error": "engine_error", "detail": str(exc)})

    if settings.record:
        recorder = Recorder(settings.record)

        @app.middleware("http")
        async def _record(request: Request, call_next):
            if request.method != "POST":
                return await call_next(request)
            started = time.perf_counter()
            body = await request.body()
            response = await call_next(request)
            chunks = [chunk async for chunk in response.body_iterator]
            raw = b"".join(chunks)
            recorder.write(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "request": _json_or_text(body),
                    "response": _json_or_text(raw),
                }
            )
            return Response(
                content=raw,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

    def decider(request: Request) -> Decider:
        return request.app.state.decider

    @app.get("/health")
    async def health(request: Request) -> dict:
        d = decider(request)
        return {"status": "ok", "engine": d.engine.name, "model": d.engine.model}

    @app.post("/v1/decide", response_model=DecideResponse, response_model_exclude_none=True)
    async def decide(req: DecideRequest, request: Request):
        """Generic decision: binary, choice, and scale questions about one input."""
        return await usecases.decide(decider(request), req)

    @app.post("/v1/route", response_model=RouteResponse)
    async def route(req: RouteRequest, request: Request):
        """Pick which model (or handler) should take a request."""
        return await usecases.route(decider(request), req)

    @app.post("/v1/guard", response_model=GuardResponse)
    async def guard(req: GuardRequest, request: Request):
        """Screen input for injection, abuse, or off-topic use before it reaches an LLM."""
        return await usecases.guard(decider(request), req)

    @app.post("/v1/gate", response_model=GateResponse)
    async def gate(req: GateRequest, request: Request):
        """Decide whether an agent's tool call is allowed, needs confirmation, or is denied."""
        return await usecases.gate(decider(request), req)

    @app.post("/v1/classify", response_model=ClassifyResponse)
    async def classify(req: ClassifyRequest, request: Request):
        """Label one item or a batch: inbox triage, intent detection, bulk labeling."""
        return await usecases.classify(decider(request), req)

    @app.post("/v1/judge", response_model=JudgeResponse)
    async def judge(req: JudgeRequest, request: Request):
        """Score an LLM output against a rubric."""
        return await usecases.judge(decider(request), req)

    @app.post("/v1/rerank", response_model=RerankResponse, response_model_exclude_none=True)
    @app.post("/v2/rerank", response_model=RerankResponse, response_model_exclude_none=True)
    async def rerank(req: RerankRequest, request: Request):
        """Order documents by relevance to a query. Cohere-style wire format."""
        return await usecases.rerank(decider(request), req)

    return app


def _json_or_text(raw: bytes):
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode("utf-8", errors="replace")
