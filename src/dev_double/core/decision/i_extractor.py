"""Optional capability: a decider that extracts a whole record natively (e.g.
one tool call with every field) instead of one question per field.

The extract use case checks ``isinstance(decider, IExtractor)`` and uses it when present.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .t_extract import TExtractRequest, TFieldValue
from .tracker import Tracker


@runtime_checkable
class IExtractor(Protocol):
    async def extract(self, req: TExtractRequest, tracker: Tracker) -> dict[str, TFieldValue]:
        """Field name -> value. Fields left out are reported as empty."""
        ...
