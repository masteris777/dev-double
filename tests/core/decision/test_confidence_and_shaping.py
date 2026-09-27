import pytest

from dev_double.core.decision.answer_shaping import question_labels, shape_answer
from dev_double.core.decision.confidence import confidence
from dev_double.core.decision.t_answer import TBinaryAnswer, TChoiceAnswer, TScaleAnswer
from dev_double.core.decision.t_question import TBinaryQuestion, TChoiceQuestion, TScaleQuestion


def test_confidence():
    assert confidence([1.0, 0.0, 0.0]) == 1.0
    assert confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert confidence([0.8, 0.2]) == pytest.approx(0.6)
    assert confidence([1.0]) == 1.0


def test_shape_binary():
    a = shape_answer(TBinaryQuestion("q"), {"yes": 0.812345, "no": 0.187655})
    assert a == TBinaryAnswer(value=True, probability=0.8123, confidence=0.6247)


def test_shape_choice_rounds_unless_asked_not_to():
    q = TChoiceQuestion("q", {"a": "x", "b": "y"})
    a = shape_answer(q, {"a": 0.123456, "b": 0.876544})
    assert isinstance(a, TChoiceAnswer)
    assert a.value == "b" and a.probabilities == {"a": 0.1235, "b": 0.8765}
    raw = shape_answer(q, {"a": 0.123456, "b": 0.876544}, round_probs=False)
    assert raw.probabilities == {"a": 0.123456, "b": 0.876544}  # type: ignore[union-attr]


def test_shape_scale():
    q = TScaleQuestion("q", ["low", "mid", "high"])
    a = shape_answer(q, {"0": 0.1, "1": 0.3, "2": 0.6})
    assert isinstance(a, TScaleAnswer)
    assert a.value == pytest.approx(1.5) and a.level == 2
    assert a.legend == {"0": "low", "1": "mid", "2": "high"}


def test_question_labels():
    assert list(question_labels(TBinaryQuestion("q"))) == ["yes", "no"]
    assert list(question_labels(TScaleQuestion("q", ["a", "b"]))) == ["0", "1"]
