"""Deterministic ids: ``<prefix>-1``, ``<prefix>-2``, ..."""

from __future__ import annotations

from dev_double.core.decision.i_id_provider import IIdProvider


class IdProviderMockImpl(IIdProvider):
    def __init__(self, prefix: str = "mock") -> None:
        self._prefix = prefix
        self._n = 0

    def new_id(self) -> str:
        self._n += 1
        return f"{self._prefix}-{self._n}"
