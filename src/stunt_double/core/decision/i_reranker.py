"""Optional capability: a decider that scores documents natively (e.g. with
embeddings) instead of one yes/no question per document.

The rerank use case checks ``isinstance(decider, IReranker)`` and uses it when present.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .tracker import Tracker


@runtime_checkable
class IReranker(Protocol):
    async def rerank_scores(self, query: str, documents: list[str], tracker: Tracker) -> list[float]:
        """One relevance score per document, in input order; higher is more relevant."""
        ...
