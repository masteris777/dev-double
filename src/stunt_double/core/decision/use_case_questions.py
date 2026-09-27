"""The questions and payloads each use case asks. Wording was tuned by experiment."""

from __future__ import annotations

from .defaults import DEFAULT_OUTCOMES, DEFAULT_POLICIES, DEFAULT_ROUTES, DEFAULT_RUBRIC
from .t_gate import TGateRequest
from .t_guard import TGuardRequest
from .t_judge import TJudgeRequest
from .t_question import TBinaryQuestion, TChoiceQuestion, TScaleQuestion
from .t_route import TRouteRequest


def route_question(req: TRouteRequest) -> TChoiceQuestion:
    return TChoiceQuestion(
        question="Which option is the best fit for handling this request?",
        options=req.routes or DEFAULT_ROUTES,
    )


def guard_questions(req: TGuardRequest) -> dict[str, TBinaryQuestion]:
    policies = dict(req.policies or DEFAULT_POLICIES)
    if req.scope:
        policies["off_topic"] = f"The input is unrelated to this assistant's purpose: {req.scope}"
    return {
        name: TBinaryQuestion(question=f"Does the input do the following? {description}")
        for name, description in policies.items()
    }


def gate_payload(req: TGateRequest) -> dict:
    arguments = req.tool_call.arguments
    payload: dict = {
        "tool_call": {
            "name": req.tool_call.name,
            "arguments": arguments if isinstance(arguments, dict) else str(arguments),
        }
    }
    if req.context is not None:
        payload["context"] = req.context
    if req.policy:
        payload["policy"] = req.policy
    return payload


def gate_question(req: TGateRequest) -> TChoiceQuestion:
    return TChoiceQuestion(
        question=(
            "An AI agent wants to make this tool call. What should happen? "
            "Check that the call matches what the user asked for, and whether it moves money, "
            "deletes data, or is costly or irreversible."
        ),
        options=req.outcomes or DEFAULT_OUTCOMES,
    )


def judge_payload(req: TJudgeRequest) -> dict:
    # Task first, output last: the model reads what was asked before judging the answer.
    payload: dict = {}
    if req.input is not None:
        payload["task"] = req.input
    if req.reference is not None:
        payload["reference_answer"] = req.reference
    payload["output"] = req.output
    return payload


def judge_question(req: TJudgeRequest) -> TScaleQuestion:
    return TScaleQuestion(
        question=(
            f"How good is the output as an answer to the task? Criteria: {req.criteria} "
            "If a reference answer is given, an output that contradicts it is wrong."
        ),
        levels=req.levels or DEFAULT_RUBRIC,
    )


def rerank_question(query: str) -> TBinaryQuestion:
    return TBinaryQuestion(
        question=f"Is this document relevant to the query below and useful for answering it?\nQuery: {query}",
    )
