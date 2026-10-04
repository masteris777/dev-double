"""Domain errors."""

from __future__ import annotations


class EngineError(RuntimeError):
    """The backing model could not be reached or returned something unusable."""


class UnsupportedRequestError(Exception):
    """The request is valid, but the configured engine can't answer it as asked
    (for example more options than it can score). The caller can change the request."""
