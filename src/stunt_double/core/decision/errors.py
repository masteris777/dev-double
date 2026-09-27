"""Domain errors."""

from __future__ import annotations


class EngineError(RuntimeError):
    """The backing model could not be reached or returned something unusable."""
