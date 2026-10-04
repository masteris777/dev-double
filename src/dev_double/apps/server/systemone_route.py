"""HTTP behavior specific to POST /v1/systemone, which follows Ollama's error
contract: every error is ``{"error": "message"}``, validation failures are 400
(not FastAPI's 422), and oversized bodies are 413. Other routes are unaffected.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from ...core.decision.i_decision_service import IDecisionService

PATH = "/v1/systemone"
MAX_BODY_BYTES = 64 * 1024  # without images
MAX_BODY_BYTES_WITH_IMAGES = 32 * 1024 * 1024


def error_response(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


def images_unsupported(service: IDecisionService) -> str:
    """Why ``service`` can't answer a request with images."""
    message = "this engine does not support images"
    if service.name == "openai":
        message += "; start the server with --vision-model (for example qwen2.5vl:7b) to read them"
    return message


def validation_message(exc: RequestValidationError) -> str:
    """The first validation problem as one readable sentence."""
    errors = exc.errors()
    if not errors:
        return "invalid request"
    err = errors[0]
    if err.get("type") == "json_invalid":
        return "request body must be valid JSON"
    message = str(err.get("msg", "invalid request")).removeprefix("Value error, ")
    loc = ".".join(str(part) for part in err.get("loc", ()) if part != "body")
    return f"{loc}: {message}" if loc else message


def _has_images(body: bytes) -> bool:
    try:
        data: Any = json.loads(body)
    except ValueError:
        return False
    return isinstance(data, dict) and isinstance(data.get("images"), list) and bool(data["images"])


class _TooLarge(Exception):
    """The body is over the contract's limit; the message is the 413's."""


async def _read_limited(request: Request) -> bytes:
    """The request body. Raises ``_TooLarge`` as soon as the larger limit is passed,
    or when the body is over the smaller one and carries no images."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BODY_BYTES_WITH_IMAGES:
        raise _TooLarge("request body must not exceed 32 MiB with images")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES_WITH_IMAGES:
            raise _TooLarge("request body must not exceed 32 MiB with images")
        chunks.append(chunk)
    body = b"".join(chunks)
    if size > MAX_BODY_BYTES and not _has_images(body):
        raise _TooLarge("request body must not exceed 64 KiB without images")
    return body


class SystemOneRoute(APIRoute):
    """Enforces the body limits before the body is parsed."""

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        handler = super().get_route_handler()

        async def limited(request: Request) -> Response:
            try:
                body = await _read_limited(request)
            except _TooLarge as exc:
                return error_response(413, str(exc))

            async def replay() -> dict[str, Any]:
                return {"type": "http.request", "body": body, "more_body": False}

            return await handler(Request(request.scope, replay))

        return limited
