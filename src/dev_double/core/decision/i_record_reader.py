"""Optional decider capability: read the free-text fields (string, number,
integer) of an extraction request in one pass, as a whole record.

The extract use case checks ``isinstance(decider, IRecordReader)``; deciders
without it (e.g. System 1 models, which only answer typed questions) leave
text fields empty with a warning.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .t_extract import TExtractField, TFieldValue
from .t_input import TInputValue
from .tracker import Tracker


@runtime_checkable
class IRecordReader(Protocol):
    async def read_record(
        self, input_value: TInputValue, fields: dict[str, TExtractField], tracker: Tracker
    ) -> dict[str, TFieldValue]:
        """Field name -> parsed value (None when absent or unreadable) and confidence;
        add usage and warnings to ``tracker``. Raises EngineError on backend failure."""
        ...
