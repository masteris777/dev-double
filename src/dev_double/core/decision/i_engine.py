"""The LLM label-scoring port: given a rendered prompt and a fixed set of
labels, return a probability distribution over those labels.

Everything above this port (question types, use cases, confidence) is
engine-agnostic, so swapping the model is a configuration change.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .t_label_query import TLabelQuery, TLabelResult


@runtime_checkable
class IEngine(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def supports_images(self) -> bool:
        """Whether a query may carry ``images``."""
        return False

    def model_for(self, images: bool) -> str:
        """The model that answers a query with (or without) images, for response meta."""
        return self.model

    async def distribution(self, query: TLabelQuery) -> TLabelResult:
        """Raises EngineError when the model can't be reached or answers unusably."""
        ...

    async def aclose(self) -> None: ...
