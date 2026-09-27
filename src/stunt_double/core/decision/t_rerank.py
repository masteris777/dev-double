"""Order documents by relevance to a query."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .t_meta import TMeta


@dataclass(frozen=True)
class TRerankRequest:
    query: str
    documents: list[str]
    top_n: Optional[int] = None
    return_documents: bool = False

    def __post_init__(self) -> None:
        if not self.documents:
            raise ValueError("`documents` must not be empty.")
        if self.top_n is not None and self.top_n < 1:
            raise ValueError("`top_n` must be at least 1.")


@dataclass(frozen=True)
class TRerankResult:
    index: int
    relevance_score: float
    document: Optional[str] = None  # set when return_documents is true


@dataclass(frozen=True)
class TRerankResponse:
    id: str
    results: list[TRerankResult]
    meta: TMeta
