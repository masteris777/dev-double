"""Resource auto-detection: integration suites run when the resource answers,
and skip cleanly when it doesn't. No manual flags."""

from __future__ import annotations

import functools
import importlib.util

import httpx

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
VISION_MODEL = "qwen2.5vl:7b"
SYSTEMONE_URL = "http://localhost:8000"


@functools.lru_cache(maxsize=None)
def has_ollama_model(model: str = OLLAMA_MODEL) -> bool:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=1.0)
        names = {m.get("name") for m in r.json().get("models", [])}
    except (httpx.HTTPError, ValueError, AttributeError):
        return False
    return model in names


@functools.lru_cache(maxsize=None)
def has_systemone() -> bool:
    """A server answers POST /v1/systemone (any non-404, non-5xx reply counts)."""
    try:
        r = httpx.post(f"{SYSTEMONE_URL}/v1/systemone", json={}, timeout=1.0)
    except httpx.HTTPError:
        return False
    return r.status_code != 404 and r.status_code < 500


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None
