"""Randomness port: mints response ids."""

from __future__ import annotations

from typing import Protocol


class IIdProvider(Protocol):
    def new_id(self) -> str: ...
