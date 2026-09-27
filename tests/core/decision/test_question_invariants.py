import pytest

from dev_double.core.decision.t_classify import TClassifyRequest
from dev_double.core.decision.t_decide import TDecideRequest
from dev_double.core.decision.t_gate import TGateRequest, TToolCall
from dev_double.core.decision.t_guard import TGuardRequest
from dev_double.core.decision.t_judge import TJudgeRequest
from dev_double.core.decision.t_question import (
    MAX_LEVELS,
    MAX_OPTIONS,
    TChoiceQuestion,
    TScaleQuestion,
    check_option_keys,
)
from dev_double.core.decision.t_rerank import TRerankRequest
from dev_double.core.decision.t_route import TRouteRequest


def test_choice_needs_2_to_20_options():
    with pytest.raises(ValueError):
        TChoiceQuestion("q", {"a": "only one"})
    with pytest.raises(ValueError):
        TChoiceQuestion("q", {f"o{i}": "x" for i in range(MAX_OPTIONS + 1)})
    TChoiceQuestion("q", {f"o{i}": "x" for i in range(MAX_OPTIONS)})


def test_scale_needs_2_to_10_levels():
    with pytest.raises(ValueError):
        TScaleQuestion("q", ["one"])
    with pytest.raises(ValueError):
        TScaleQuestion("q", ["x"] * (MAX_LEVELS + 1))


def test_option_keys_must_differ_ignoring_case_and_punctuation():
    with pytest.raises(ValueError, match="too similar"):
        check_option_keys({"reply_now": "a", "Reply-Now": "b"})
    with pytest.raises(ValueError, match="letter or digit"):
        check_option_keys({"--": "a", "b": "b"})
    with pytest.raises(ValueError, match="too similar"):
        TChoiceQuestion("q", {"yes": "a", "YES!": "b"})


def test_use_case_requests_enforce_invariants():
    with pytest.raises(ValueError):
        TDecideRequest("x", {})
    with pytest.raises(ValueError, match="too similar"):
        TRouteRequest("x", routes={"a_b": "1", "ab": "2"})
    with pytest.raises(ValueError):
        TGateRequest(TToolCall("t"), outcomes={"only": "one"})
    with pytest.raises(ValueError):
        TGuardRequest("x", threshold=1.5)
    with pytest.raises(ValueError):
        TGuardRequest("x", policies={})
    with pytest.raises(ValueError, match="exactly one"):
        TClassifyRequest({"a": "x", "b": "y"})
    with pytest.raises(ValueError, match="exactly one"):
        TClassifyRequest({"a": "x", "b": "y"}, input="i", inputs=["j"])
    with pytest.raises(ValueError, match="empty"):
        TClassifyRequest({"a": "x", "b": "y"}, inputs=[])
    with pytest.raises(ValueError):
        TJudgeRequest("out", levels=["one"])
    with pytest.raises(ValueError):
        TRerankRequest("q", [])
    with pytest.raises(ValueError):
        TRerankRequest("q", ["d"], top_n=0)


def test_classify_items():
    assert TClassifyRequest({"a": "x", "b": "y"}, input="i").items() == ["i"]
    assert TClassifyRequest({"a": "x", "b": "y"}, inputs=["i", "j"]).items() == ["i", "j"]
