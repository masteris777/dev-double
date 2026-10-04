"""The System One wire format (``POST /v1/systemone``), in both directions.

The server endpoint reads requests and writes answers with this module; the
System 1 decider writes requests and reads answers with it. Both sides share
one mapping, so what dev-double serves and what it consumes can't drift apart.

    wire question   <->  core question
    noul            <->  TBinaryQuestion   (criteria "true"/"false" <-> yes/no)
    choice          <->  TChoiceQuestion   (criteria option -> description or null)
    score           <->  TScaleQuestion    (criteria list, lowest first)

Every function raises ``ValueError`` (with the offending path in the message)
when the wire data breaks the contract; callers turn that into their own error.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from .answer_shaping import question_labels
from .confidence import DIGITS, entropy_confidence
from .images import check_image
from .t_answer import TAnswer, TBinaryAnswer, TChoiceAnswer
from .t_input import TInputValue
from .t_question import (
    MAX_LEVELS,
    MAX_OPTIONS,
    TBinaryQuestion,
    TChoiceQuestion,
    TQuestion,
    TScaleQuestion,
)
from .t_usage import TUsage

MAX_QUESTIONS = 64  # per request
QUESTION_TYPES = ("choice", "noul", "score")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def content_text(value: Any, where: str) -> str:
    """Instructions as text: a non-blank string, or an object or array as JSON text."""
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"`{where}` must not be blank.")
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    raise ValueError(f"`{where}` must be a string, object, or array.")


def state_from_wire(state: Any) -> TInputValue:
    """The shared content the questions are about: a non-blank string, an object, or an array."""
    if isinstance(state, (dict, list)):
        return state
    return content_text(state, "state")


def _nonblank_key(key: Any, where: str) -> str:
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"`{where}` keys must not be blank.")
    return key


def _criteria_size(count: int, limit: int, where: str) -> None:
    if not 2 <= count <= limit:
        raise ValueError(f"`{where}` must have 2 to {limit} entries, got {count}.")


def question_from_wire(q: Any, where: str = "question") -> TQuestion:
    """A wire question as a core question."""
    if not isinstance(q, dict):
        raise ValueError(f"`{where}` must be an object.")
    kind = q.get("type")
    if kind not in QUESTION_TYPES:
        raise ValueError(f"`{where}.type` must be one of {', '.join(QUESTION_TYPES)}; got {kind!r}.")
    text = content_text(q.get("instructions"), f"{where}.instructions")
    criteria = q.get("criteria")
    at = f"{where}.criteria"

    if kind == "noul":
        if criteria is None:
            return TBinaryQuestion(question=text)
        if not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}:
            raise ValueError(f"`{at}` may only have the keys \"true\" and \"false\".")
        for key, description in criteria.items():
            if description is not None and not isinstance(description, str):
                raise ValueError(f"`{at}.{key}` must be a string.")
        return TBinaryQuestion(question=text, yes=criteria.get("true") or None, no=criteria.get("false") or None)

    if kind == "choice":
        if not isinstance(criteria, dict):
            raise ValueError(f"`{at}` must be an object of option -> description.")
        _criteria_size(len(criteria), MAX_OPTIONS, at)
        options: dict[str, str] = {}
        for key, description in criteria.items():
            _nonblank_key(key, at)
            if description is not None and not isinstance(description, str):
                raise ValueError(f"`{at}.{key}` must be a string or null.")
            options[key] = key if description is None else description  # null: the key describes itself
        try:
            return TChoiceQuestion(question=text, options=options)
        except ValueError as exc:
            raise ValueError(f"`{at}`: {exc}") from exc

    if not isinstance(criteria, list) or not all(isinstance(item, str) for item in criteria):
        raise ValueError(f"`{at}` must be a list of strings, lowest score first.")
    _criteria_size(len(criteria), MAX_LEVELS, at)
    return TScaleQuestion(question=text, levels=list(criteria))


def question_to_wire(q: TQuestion) -> dict[str, Any]:
    """A core question as a wire question."""
    if isinstance(q, TBinaryQuestion):
        body: dict[str, Any] = {"type": "noul", "instructions": q.question}
        criteria = {k: v for k, v in (("true", q.yes), ("false", q.no)) if v}
        if criteria:
            body["criteria"] = criteria
        return body
    if isinstance(q, TChoiceQuestion):
        return {"type": "choice", "instructions": q.question, "criteria": dict(q.options)}
    return {"type": "score", "instructions": q.question, "criteria": list(q.levels)}


def questions_from_wire(questions: Any) -> dict[str, TQuestion]:
    """The named questions of a request, 1 to MAX_QUESTIONS of them."""
    if not isinstance(questions, dict):
        raise ValueError("`questions` must be an object of name -> question.")
    if not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ValueError(f"`questions` must have 1 to {MAX_QUESTIONS} entries, got {len(questions)}.")
    return {
        _nonblank_key(name, "questions"): question_from_wire(q, f"questions.{name}")
        for name, q in questions.items()
    }


def images_from_wire(images: Any) -> tuple[str, ...]:
    """The request's images: plain base64 PNG, JPEG, or WebP, in request order."""
    if images is None:
        return ()
    if not isinstance(images, list):
        raise ValueError("`images` must be an array of base64 strings.")
    for i, image in enumerate(images):
        if not isinstance(image, str):
            raise ValueError(f"`images[{i}]` must be a base64 string.")
        check_image(image, f"images[{i}]")
    return tuple(images)


def answer_to_wire(a: TAnswer) -> dict[str, Any]:
    """A core answer as a wire answer. A noul answer is the probability of true.
    Confidence is recomputed with the wire's formula (normalized entropy)."""
    if isinstance(a, TBinaryAnswer):
        return {"type": "noul", "noul": a.probability}
    conf = round(entropy_confidence(list(a.probabilities.values())), DIGITS)
    if isinstance(a, TChoiceAnswer):
        return {
            "type": "choice",
            "choice": a.value,
            "probabilities": dict(a.probabilities),
            "confidence": conf,
        }
    return {
        "type": "score",
        "score": a.value,
        "legend": dict(a.legend),
        "probabilities": dict(a.probabilities),
        "confidence": conf,
    }


def probabilities_from_wire(q: TQuestion, answer: Any, where: str = "answer") -> dict[str, float]:
    """A wire answer as a distribution over the question's labels (yes/no, option
    keys, or "0".."n-1"), in the question's label order."""
    if not isinstance(answer, dict):
        raise ValueError(f"`{where}` must be an object.")
    if isinstance(q, TBinaryQuestion):
        p = answer.get("noul")
        if not _is_number(p) or not 0 <= p <= 1:
            raise ValueError(f"`{where}.noul` must be a number from 0 to 1.")
        return {"yes": float(p), "no": 1 - float(p)}
    probs = answer.get("probabilities")
    labels = list(question_labels(q))
    if not isinstance(probs, dict) or set(probs) != set(labels):
        raise ValueError(f"`{where}.probabilities` must have exactly the keys {labels}.")
    if not all(_is_number(v) and v >= 0 for v in probs.values()):
        raise ValueError(f"`{where}.probabilities` must be non-negative numbers.")
    return {label: float(probs[label]) for label in labels}


def usage_to_wire(usage: TUsage) -> dict[str, int]:
    return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}


def usage_from_wire(usage: Optional[dict[str, Any]]) -> TUsage:
    usage = usage or {}
    return TUsage(input_tokens=int(usage.get("input_tokens", 0)), output_tokens=int(usage.get("output_tokens", 0)))


def request_to_wire(
    model: str, state: TInputValue, questions: dict[str, TQuestion], images: Sequence[str] = ()
) -> dict[str, Any]:
    """The request body for ``questions`` about ``state`` (and ``images``, when there are any)."""
    body: dict[str, Any] = {
        "model": model,
        "state": state,
        "questions": {name: question_to_wire(q) for name, q in questions.items()},
    }
    if images:
        body["images"] = list(images)
    return body


def response_to_wire(model: str, answers: dict[str, TAnswer], usage: TUsage) -> dict[str, Any]:
    """The response body: the request's ``model`` echoed, one answer per question name."""
    return {
        "model": model,
        "answers": {name: answer_to_wire(a) for name, a in answers.items()},
        "usage": usage_to_wire(usage),
    }
