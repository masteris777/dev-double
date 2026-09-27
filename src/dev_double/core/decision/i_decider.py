"""Answers one typed question about one input.

``DeciderBasicImpl`` does it by prompting an ``IEngine``; decision models that
answer typed questions natively (Kev, Laya, Needle) implement this directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .t_answer import TAnswer
from .t_input import TInputValue
from .t_question import TQuestion
from .tracker import Tracker


@runtime_checkable
class IDecider(Protocol):
    @property
    def name(self) -> str:
        """Engine name, reported in response meta."""
        ...

    @property
    def model(self) -> str:
        """Model name, reported in response meta."""
        ...

    async def ask(
        self, value: TInputValue, question: TQuestion, tracker: Tracker, label: str = ""
    ) -> TAnswer:
        """Answer ``question`` about ``value``; add usage and warnings (prefixed
        with ``label``) to ``tracker``. Raises EngineError on backend failure."""
        ...

    async def aclose(self) -> None: ...
