"""Answers one typed question about one input.

``DeciderBasicImpl`` does it by prompting an ``IEngine``; decision models that
answer typed questions natively (Kev, Laya, Needle) implement this directly.
"""

from __future__ import annotations

import asyncio
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

    @property
    def supports_images(self) -> bool:
        """Whether ``images`` may be passed to ``ask`` and ``ask_many``."""
        return False

    def model_for(self, images: bool) -> str:
        """The model that answers a request with (or without) images, for response meta."""
        return self.model

    async def ask(
        self,
        value: TInputValue,
        question: TQuestion,
        tracker: Tracker,
        label: str = "",
        images: tuple[str, ...] = (),
    ) -> TAnswer:
        """Answer ``question`` about ``value`` and ``images`` (base64 PNG, JPEG, or WebP;
        only when ``supports_images``); add usage and warnings (prefixed with ``label``)
        to ``tracker``. Raises EngineError on backend failure."""
        ...

    async def ask_many(
        self,
        value: TInputValue,
        questions: dict[str, TQuestion],
        tracker: Tracker,
        label: str = "",
        images: tuple[str, ...] = (),
    ) -> dict[str, TAnswer]:
        """Answer several questions about the same ``value`` and ``images``, keyed by name.

        The default asks them concurrently; a decider whose backend takes many
        questions per call overrides this to send them together. Warnings are
        prefixed with the question name, after ``label`` if given."""
        prefix = f"{label}." if label else ""
        answers = await asyncio.gather(
            *(
                self.ask(value, q, tracker, label=f"{prefix}{name}", images=images)
                for name, q in questions.items()
            )
        )
        return dict(zip(questions, answers))

    async def aclose(self) -> None: ...
