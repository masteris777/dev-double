"""Decision core: interfaces, domain types, prompts, and the use cases.

Zero third-party dependencies. Side effects (model calls, time, ids) arrive
through the ports ``IEngine`` / ``IDecider``, ``IClock``, and ``IIdProvider``.
Optional capabilities (``IGenerator``, ``IRecordReader``, ``IExtractor``,
``IReranker``) are detected with ``isinstance`` and used when present.
"""

from .confidence import DIGITS, confidence
from .decider_basic_impl import DeciderBasicImpl
from .decision_service_basic_impl import DecisionServiceBasicImpl
from .defaults import DEFAULT_OUTCOMES, DEFAULT_POLICIES, DEFAULT_ROUTES, DEFAULT_RUBRIC
from .errors import EngineError, UnsupportedRequestError
from .i_clock import IClock
from .i_decider import IDecider
from .i_decision_service import IDecisionService
from .i_engine import IEngine
from .i_extractor import IExtractor
from .i_generator import IGenerator
from .i_id_provider import IIdProvider
from .i_record_reader import IRecordReader
from .i_reranker import IReranker
from .t_answer import TAnswer, TBinaryAnswer, TChoiceAnswer, TScaleAnswer
from .t_classify import TClassification, TClassifyRequest, TClassifyResponse
from .t_decide import TDecideRequest, TDecideResponse
from .t_extract import MAX_FIELDS, TExtractField, TExtractRequest, TExtractResponse, TFieldValue
from .t_gate import TGateRequest, TGateResponse, TToolCall
from .t_generate import TGenerateQuery, TGenerateResult
from .t_guard import TGuardCheck, TGuardRequest, TGuardResponse
from .t_input import TInputValue
from .t_judge import TJudgeRequest, TJudgeResponse
from .t_label_query import TLabelQuery, TLabelResult, TPosition
from .t_meta import TMeta
from .t_question import (
    MAX_LEVELS,
    MAX_OPTIONS,
    TBinaryQuestion,
    TChoiceQuestion,
    TQuestion,
    TScaleQuestion,
)
from .t_rerank import TRerankRequest, TRerankResponse, TRerankResult
from .t_route import TRouteRequest, TRouteResponse
from .t_usage import TUsage
from .tracker import Tracker

__all__ = [
    "DEFAULT_OUTCOMES", "DEFAULT_POLICIES", "DEFAULT_ROUTES", "DEFAULT_RUBRIC", "DIGITS",
    "MAX_FIELDS", "MAX_LEVELS", "MAX_OPTIONS",
    "DeciderBasicImpl", "DecisionServiceBasicImpl", "EngineError", "UnsupportedRequestError",
    "IClock", "IDecider", "IDecisionService", "IEngine", "IExtractor",
    "IGenerator", "IIdProvider", "IRecordReader", "IReranker",
    "TExtractField", "TExtractRequest", "TExtractResponse", "TFieldValue",
    "TGenerateQuery", "TGenerateResult",
    "TAnswer", "TBinaryAnswer", "TBinaryQuestion", "TChoiceAnswer", "TChoiceQuestion",
    "TClassification", "TClassifyRequest", "TClassifyResponse", "TDecideRequest",
    "TDecideResponse", "TGateRequest", "TGateResponse", "TGuardCheck", "TGuardRequest",
    "TGuardResponse", "TInputValue", "TJudgeRequest", "TJudgeResponse", "TLabelQuery",
    "TLabelResult", "TMeta", "TPosition", "TQuestion", "TRerankRequest", "TRerankResponse",
    "TRerankResult", "TRouteRequest", "TRouteResponse", "TScaleAnswer", "TScaleQuestion",
    "TToolCall", "TUsage", "Tracker", "confidence",
]
