"""Optional engine capability: generate a short free-text answer.

Label scoring (``IEngine.distribution``) can't produce values that aren't known
up front, such as a vendor name or an amount. Engines that can also generate
text implement this; ``DeciderBasicImpl`` checks ``isinstance(engine, IGenerator)``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .t_generate import TGenerateQuery, TGenerateResult


@runtime_checkable
class IGenerator(Protocol):
    async def generate(self, query: TGenerateQuery) -> TGenerateResult:
        """Raises EngineError when the model can't be reached or answers unusably."""
        ...
