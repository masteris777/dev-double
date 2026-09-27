"""Turns a distribution over a question's labels into its typed answer.

Labels are "yes"/"no" for binary questions, option keys for choice questions,
and "0".."n-1" for scale questions (see ``question_labels``).
"""

from __future__ import annotations

from .confidence import DIGITS, confidence, round_probabilities
from .t_answer import TAnswer, TBinaryAnswer, TChoiceAnswer, TScaleAnswer
from .t_question import TBinaryQuestion, TChoiceQuestion, TQuestion


def question_labels(question: TQuestion) -> dict[str, str]:
    """Label -> description, in the order the question defines them."""
    if isinstance(question, TBinaryQuestion):
        return {"yes": question.yes or "The answer is yes.", "no": question.no or "The answer is no."}
    if isinstance(question, TChoiceQuestion):
        return dict(question.options)
    return {str(i): text for i, text in enumerate(question.levels)}


def shape_answer(question: TQuestion, probs: dict[str, float], round_probs: bool = True) -> TAnswer:
    """``round_probs=False`` passes choice/scale probabilities through as given."""
    if isinstance(question, TBinaryQuestion):
        p_yes = probs["yes"]
        return TBinaryAnswer(
            value=p_yes >= 0.5,
            probability=round(p_yes, DIGITS),
            confidence=round(confidence([p_yes, 1 - p_yes]), DIGITS),
        )

    shown = round_probabilities(probs) if round_probs else probs
    if isinstance(question, TChoiceQuestion):
        return TChoiceAnswer(
            value=max(probs, key=probs.__getitem__),
            probabilities=shown,
            confidence=round(confidence(list(probs.values())), DIGITS),
        )

    expected = sum(int(level) * p for level, p in probs.items())
    return TScaleAnswer(
        value=round(expected, DIGITS),
        level=int(max(probs, key=probs.__getitem__)),
        probabilities=shown,
        legend={str(i): text for i, text in enumerate(question.levels)},
        confidence=round(confidence(list(probs.values())), DIGITS),
    )
