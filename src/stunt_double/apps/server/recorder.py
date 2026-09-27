"""Appends every API call to a JSONL file, so you can replay the same
traffic against the real engine later and compare."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import Response


class Recorder:
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()

    def write(self, entry: dict) -> None:
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock, open(self._path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def install_recorder(app: FastAPI, path: str) -> None:
    """Record every POST: request, response, status, and latency."""
    recorder = Recorder(path)

    @app.middleware("http")
    async def _record(request: Request, call_next):  # type: ignore[no-untyped-def]
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


def _json_or_text(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode("utf-8", errors="replace")
