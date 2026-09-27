"""The HTTP server: maps transport schemas to the core use-case service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ... import __version__
from ...core.decision.errors import EngineError
from ...core.decision.i_decider import IDecider
from ...core.decision.i_decision_service import IDecisionService
from ...core.decision.i_engine import IEngine
from ..composition import build_service
from ..config import Settings
from .recorder import install_recorder
from .transport import (
    ClassifyRequest,
    ClassifyResponse,
    DecideRequest,
    DecideResponse,
    ExtractRequest,
    ExtractResponse,
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


def create_app(
    settings: Optional[Settings] = None,
    engine: Optional[IEngine] = None,
    decider: Optional[IDecider] = None,
) -> FastAPI:
    """``engine`` / ``decider`` override the one ``settings`` would build."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        svc = build_service(settings, engine=engine, decider=decider)
        app.state.service = svc
        try:
            yield
        finally:
            await svc.aclose()

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
        install_recorder(app, settings.record)

    def service(request: Request) -> IDecisionService:
        return request.app.state.service

    @app.get("/health")
    async def health(request: Request) -> dict:
        svc = service(request)
        return {"status": "ok", "engine": svc.name, "model": svc.model}

    @app.post("/v1/decide", response_model=DecideResponse, response_model_exclude_none=True)
    async def decide(req: DecideRequest, request: Request):  # type: ignore[no-untyped-def]
        """Generic decision: binary, choice, and scale questions about one input."""
        return DecideResponse.from_core(await service(request).decide(req.to_core()))

    @app.post("/v1/route", response_model=RouteResponse)
    async def route(req: RouteRequest, request: Request):  # type: ignore[no-untyped-def]
        """Pick which model (or handler) should take a request."""
        return RouteResponse.from_core(await service(request).route(req.to_core()))

    @app.post("/v1/guard", response_model=GuardResponse)
    async def guard(req: GuardRequest, request: Request):  # type: ignore[no-untyped-def]
        """Screen input for injection, abuse, or off-topic use before it reaches an LLM."""
        return GuardResponse.from_core(await service(request).guard(req.to_core()))

    @app.post("/v1/gate", response_model=GateResponse)
    async def gate(req: GateRequest, request: Request):  # type: ignore[no-untyped-def]
        """Decide whether an agent's tool call is allowed, needs confirmation, or is denied."""
        return GateResponse.from_core(await service(request).gate(req.to_core()))

    @app.post("/v1/classify", response_model=ClassifyResponse)
    async def classify(req: ClassifyRequest, request: Request):  # type: ignore[no-untyped-def]
        """Label one item or a batch: inbox triage, intent detection, bulk labeling."""
        return ClassifyResponse.from_core(await service(request).classify(req.to_core()))

    @app.post("/v1/judge", response_model=JudgeResponse)
    async def judge(req: JudgeRequest, request: Request):  # type: ignore[no-untyped-def]
        """Score an LLM output against a rubric."""
        return JudgeResponse.from_core(await service(request).judge(req.to_core()))

    @app.post("/v1/extract", response_model=ExtractResponse, response_model_exclude_none=True)
    async def extract(req: ExtractRequest, request: Request):  # type: ignore[no-untyped-def]
        """Pull typed fields out of text: invoices, tickets, bookings, tool-call arguments."""
        return ExtractResponse.from_core(await service(request).extract(req.to_core()))

    @app.post("/v1/rerank", response_model=RerankResponse, response_model_exclude_none=True)
    @app.post("/v2/rerank", response_model=RerankResponse, response_model_exclude_none=True)
    async def rerank(req: RerankRequest, request: Request):  # type: ignore[no-untyped-def]
        """Order documents by relevance to a query. Cohere-style wire format."""
        return RerankResponse.from_core(await service(request).rerank(req.to_core()))

    return app
