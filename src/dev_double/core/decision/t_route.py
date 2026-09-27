"""Model routing: which model (or handler) should take a request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .t_input import TInputValue
from .t_meta import TMeta
from .t_question import check_options


@dataclass(frozen=True)
class TRouteRequest:
    input: TInputValue
    routes: Optional[dict[str, str]] = None  # route -> when to use it; None = DEFAULT_ROUTES

    def __post_init__(self) -> None:
        if self.routes is not None:
            check_options(self.routes, "routes")


@dataclass(frozen=True)
class TRouteResponse:
    route: str
    probabilities: dict[str, float]
    confidence: float
    meta: TMeta
