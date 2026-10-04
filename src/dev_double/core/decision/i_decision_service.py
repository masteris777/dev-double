"""The task-shaped use cases, built on the three question types."""

from __future__ import annotations

from typing import Protocol

from .t_classify import TClassifyRequest, TClassifyResponse
from .t_decide import TDecideRequest, TDecideResponse
from .t_extract import TExtractRequest, TExtractResponse
from .t_gate import TGateRequest, TGateResponse
from .t_guard import TGuardRequest, TGuardResponse
from .t_judge import TJudgeRequest, TJudgeResponse
from .t_rerank import TRerankRequest, TRerankResponse
from .t_route import TRouteRequest, TRouteResponse


class IDecisionService(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def supports_images(self) -> bool:
        """Whether ``decide`` accepts images."""
        ...

    async def decide(self, req: TDecideRequest) -> TDecideResponse: ...

    async def route(self, req: TRouteRequest) -> TRouteResponse: ...

    async def guard(self, req: TGuardRequest) -> TGuardResponse: ...

    async def gate(self, req: TGateRequest) -> TGateResponse: ...

    async def classify(self, req: TClassifyRequest) -> TClassifyResponse: ...

    async def judge(self, req: TJudgeRequest) -> TJudgeResponse: ...

    async def rerank(self, req: TRerankRequest) -> TRerankResponse: ...

    async def extract(self, req: TExtractRequest) -> TExtractResponse: ...

    async def aclose(self) -> None: ...
