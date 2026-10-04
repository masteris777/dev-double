"""The HTTP server: maps transport schemas to the core use-case service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ... import __version__
from ...core.decision.errors import EngineError, UnsupportedRequestError
from ...core.decision.i_decider import IDecider
from ...core.decision.i_decision_service import IDecisionService
from ...core.decision.i_engine import IEngine
from ..composition import build_service
from ..config import Settings
from .model_services import ModelServices, model_warning
from .recorder import install_recorder
from .systemone_route import (
    PATH as SYSTEMONE_PATH,
    SystemOneRoute,
    error_response,
    images_unsupported,
    validation_message,
)
from .transport import (
    ClassifyRequest,
    ClassifyResponse,
    DecideRequest,
    DecideResponse,
    ErrorResponse,
    ExtractRequest,
    ExtractResponse,
    GateRequest,
    GateResponse,
    GuardRequest,
    GuardResponse,
    JudgeRequest,
    JudgeResponse,
    Meta,
    RerankRequest,
    RerankResponse,
    RouteRequest,
    RouteResponse,
    SystemOneRequest,
    SystemOneResponse,
)

DECIDE_DEPRECATION = "/v1/decide is deprecated and will be removed in 0.3; use /v1/systemone"


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
        injected = engine is not None or decider is not None
        app.state.models = ModelServices(settings, svc, can_build=not injected)
        try:
            yield
        finally:
            await app.state.models.aclose()  # also closes `svc`

    app = FastAPI(
        title="dev-double",
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

    @app.exception_handler(UnsupportedRequestError)
    async def _unsupported_request(_: Request, exc: UnsupportedRequestError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": "unsupported_request", "detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> Response:
        if request.url.path == SYSTEMONE_PATH:  # Ollama's contract: 400 and {"error": ...}
            return error_response(400, validation_message(exc))
        return await request_validation_exception_handler(request, exc)

    if settings.record:
        install_recorder(app, settings.record)

    def service(request: Request) -> IDecisionService:
        return request.app.state.service

    def models(request: Request) -> ModelServices:
        return request.app.state.models

    def note_model(meta: Meta, requested: Optional[str]) -> None:
        """With honor_request_model, say so when the named model couldn't be used."""
        if settings.honor_request_model:
            warning = model_warning(requested, meta.model, honor=True)
            if warning:
                meta.warnings.append(warning)

    @app.get("/health")
    async def health(request: Request) -> dict:
        svc = service(request)
        return {"status": "ok", "engine": svc.name, "model": svc.model}

    @app.post("/v1/decide", response_model=DecideResponse, response_model_exclude_none=True, deprecated=True)
    async def decide(req: DecideRequest, request: Request, response: Response):  # type: ignore[no-untyped-def]
        """Deprecated: use /v1/systemone, the same thing with the System One field names.

        Generic decision: binary, choice, and scale questions about one input."""
        response.headers["Deprecation"] = "true"
        async with models(request).use(req.model) as svc:
            out = DecideResponse.from_core(await svc.decide(req.to_core()))
        note_model(out.meta, req.model)
        out.meta.warnings.insert(0, DECIDE_DEPRECATION)
        return out

    systemone_router = APIRouter(route_class=SystemOneRoute)

    @systemone_router.post(
        "/v1/systemone",
        response_model=SystemOneResponse,
        responses={
            400: {"model": ErrorResponse, "description": "Invalid request."},
            413: {"model": ErrorResponse, "description": "Body over 64 KiB without images or 32 MiB with."},
            500: {"model": ErrorResponse, "description": "The model failed to answer."},
        },
    )
    async def systemone(req: SystemOneRequest, request: Request):  # type: ignore[no-untyped-def]
        """Answer choice, yes/no (noul), and score questions about one shared state, in one call.

        Follows Ollama's /v1/systemone contract, plus a `meta` field."""
        try:
            async with models(request).use(req.model) as svc:
                if req.images and not svc.supports_images:
                    return error_response(400, images_unsupported(svc))
                result = await svc.decide(req.to_core())
        except UnsupportedRequestError as exc:
            return error_response(400, str(exc))
        except EngineError as exc:
            return error_response(500, str(exc))
        out = SystemOneResponse.from_core(req.model, result)
        warning = model_warning(req.model, out.meta.model, honor=settings.honor_request_model)
        if warning:
            out.meta.warnings.append(warning)
        return out

    app.include_router(systemone_router)

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
        async with models(request).use(req.model) as svc:
            out = RerankResponse.from_core(await svc.rerank(req.to_core()))
        note_model(out.meta, req.model)
        return out

    return app
