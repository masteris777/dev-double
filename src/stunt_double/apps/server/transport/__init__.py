"""HTTP transport schemas: pydantic models that validate requests, describe the
OpenAPI schema, and map to and from the core T-types. All serialization lives here."""

from .choice_use_cases import (
    Classification,
    ClassifyRequest,
    ClassifyResponse,
    GateRequest,
    GateResponse,
    RouteRequest,
    RouteResponse,
    ToolCall,
)
from .common import InputValue, Meta, Usage
from .decide import (
    Answer,
    BinaryAnswer,
    BinaryQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    DecideRequest,
    DecideResponse,
    Question,
    ScaleAnswer,
    ScaleQuestion,
)
from .guard_judge import GuardCheck, GuardRequest, GuardResponse, JudgeRequest, JudgeResponse
from .rerank import RerankDocument, RerankRequest, RerankResponse, RerankResult

__all__ = [
    "Answer", "BinaryAnswer", "BinaryQuestion", "ChoiceAnswer", "ChoiceQuestion",
    "Classification", "ClassifyRequest", "ClassifyResponse", "DecideRequest", "DecideResponse",
    "GateRequest", "GateResponse", "GuardCheck", "GuardRequest", "GuardResponse", "InputValue",
    "JudgeRequest", "JudgeResponse", "Meta", "Question", "RerankDocument", "RerankRequest",
    "RerankResponse", "RerankResult", "RouteRequest", "RouteResponse", "ScaleAnswer",
    "ScaleQuestion", "ToolCall", "Usage",
]
