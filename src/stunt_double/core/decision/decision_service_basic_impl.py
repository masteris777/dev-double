"""The use cases over an ``IDecider``, with time and ids injected."""

from __future__ import annotations

import asyncio

from . import use_case_questions as uq
from .i_clock import IClock
from .i_decider import IDecider
from .i_decision_service import IDecisionService
from .i_id_provider import IIdProvider
from .i_reranker import IReranker
from .t_answer import TAnswer, TBinaryAnswer, TChoiceAnswer, TScaleAnswer
from .t_classify import TClassification, TClassifyRequest, TClassifyResponse
from .t_decide import TDecideRequest, TDecideResponse
from .t_gate import TGateRequest, TGateResponse
from .t_guard import TGuardCheck, TGuardRequest, TGuardResponse
from .t_input import TInputValue
from .t_judge import TJudgeRequest, TJudgeResponse
from .t_meta import TMeta
from .t_question import TChoiceQuestion, TQuestion
from .t_rerank import TRerankRequest, TRerankResponse, TRerankResult
from .t_route import TRouteRequest, TRouteResponse
from .tracker import Tracker


class DecisionServiceBasicImpl(IDecisionService):
    def __init__(self, decider: IDecider, clock: IClock, ids: IIdProvider) -> None:
        self.decider = decider
        self._clock = clock
        self._ids = ids

    @property
    def name(self) -> str:
        return self.decider.name

    @property
    def model(self) -> str:
        return self.decider.model

    async def aclose(self) -> None:
        await self.decider.aclose()

    def _tracker(self) -> Tracker:
        return Tracker(self._clock)

    def _meta(self, t: Tracker) -> TMeta:
        return t.meta(self.decider.name, self.decider.model)

    async def _ask_many(
        self, value: TInputValue, questions: dict[str, TQuestion], t: Tracker
    ) -> dict[str, TAnswer]:
        answers = await asyncio.gather(
            *(self.decider.ask(value, q, t, label=qid) for qid, q in questions.items())
        )
        return dict(zip(questions, answers))

    async def decide(self, req: TDecideRequest) -> TDecideResponse:
        t = self._tracker()
        answers = await self._ask_many(req.input, req.questions, t)
        return TDecideResponse(answers=answers, meta=self._meta(t))

    async def route(self, req: TRouteRequest) -> TRouteResponse:
        t = self._tracker()
        a = await self.decider.ask(req.input, uq.route_question(req), t)
        assert isinstance(a, TChoiceAnswer)
        return TRouteResponse(a.value, a.probabilities, a.confidence, self._meta(t))

    async def guard(self, req: TGuardRequest) -> TGuardResponse:
        t = self._tracker()
        answers = await self._ask_many(req.input, uq.guard_questions(req), t)
        checks = {}
        for name, a in answers.items():
            assert isinstance(a, TBinaryAnswer)
            checks[name] = TGuardCheck(probability=a.probability, flagged=a.probability >= req.threshold)
        flagged = [name for name, c in checks.items() if c.flagged]
        return TGuardResponse(allowed=not flagged, flagged=flagged, checks=checks, meta=self._meta(t))

    async def gate(self, req: TGateRequest) -> TGateResponse:
        t = self._tracker()
        a = await self.decider.ask(uq.gate_payload(req), uq.gate_question(req), t)
        assert isinstance(a, TChoiceAnswer)
        return TGateResponse(a.value, a.probabilities, a.confidence, self._meta(t))

    async def classify(self, req: TClassifyRequest) -> TClassifyResponse:
        t = self._tracker()
        q = TChoiceQuestion(question=req.question, options=req.labels)
        answers = await asyncio.gather(
            *(self.decider.ask(item, q, t, label=f"inputs[{i}]") for i, item in enumerate(req.items()))
        )
        results = []
        for a in answers:
            assert isinstance(a, TChoiceAnswer)
            results.append(TClassification(a.value, a.probabilities, a.confidence))
        return TClassifyResponse(results=results, meta=self._meta(t))

    async def judge(self, req: TJudgeRequest) -> TJudgeResponse:
        t = self._tracker()
        q = uq.judge_question(req)
        a = await self.decider.ask(uq.judge_payload(req), q, t)
        assert isinstance(a, TScaleAnswer)
        return TJudgeResponse(
            score=a.value,
            normalized=round(a.value / (len(q.levels) - 1), 4),
            level=a.level,
            probabilities=a.probabilities,
            legend=a.legend,
            confidence=a.confidence,
            meta=self._meta(t),
        )

    async def rerank(self, req: TRerankRequest) -> TRerankResponse:
        t = self._tracker()
        texts = req.documents
        scores = await self._rerank_scores(req.query, texts, t)
        order = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)
        if req.top_n is not None:
            order = order[: req.top_n]
        results = [
            TRerankResult(i, scores[i], texts[i] if req.return_documents else None) for i in order
        ]
        return TRerankResponse(id=self._ids.new_id(), results=results, meta=self._meta(t))

    async def _rerank_scores(self, query: str, texts: list[str], t: Tracker) -> list[float]:
        if isinstance(self.decider, IReranker):
            return await self.decider.rerank_scores(query, texts, t)  # e.g. embeddings
        q = uq.rerank_question(query)

        async def score(i: int, text: str) -> float:
            a = await self.decider.ask(text, q, t, label=f"documents[{i}]")
            assert isinstance(a, TBinaryAnswer)
            return a.probability

        return list(await asyncio.gather(*(score(i, text) for i, text in enumerate(texts))))
