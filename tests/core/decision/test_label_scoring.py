import math

import pytest

from stunt_double.core.decision.label_scoring import (
    label_distribution,
    label_from_text,
    normalize,
    one_hot,
    uniform,
)
from stunt_double.core.decision.t_label_query import TPosition


def lp(p: float) -> float:
    return math.log(p)


def pos(token: str, *candidates: tuple[str, float]) -> TPosition:
    return TPosition(token, [(t, lp(p)) for t, p in candidates])


def test_merges_case_and_whitespace_variants():
    positions = [pos("yes", ("yes", 0.5), (" Yes", 0.2), ("No", 0.1), ("Maybe", 0.2))]
    dist, mass = label_distribution(positions, ["yes", "no"])
    assert mass == pytest.approx(0.8)
    assert dist["yes"] == pytest.approx(0.875)
    assert dist["no"] == pytest.approx(0.125)


def test_matches_option_names_ignoring_punctuation_and_separators():
    positions = [pos("Reply", ("Reply", 0.7), ("archive.", 0.2), ("later", 0.1))]
    dist, _ = label_distribution(positions, ["reply_now", "later", "archive"])
    assert dist == pytest.approx({"reply_now": 0.7, "later": 0.1, "archive": 0.2})


def test_shared_first_token_is_split_by_next_position():
    # "re" fits both refund and rebooking; the next token decides.
    positions = [
        pos("re", ("re", 0.9), ("information", 0.1)),
        pos("booking", ("booking", 0.75), ("fund", 0.25)),
    ]
    dist, mass = label_distribution(positions, ["refund", "rebooking", "information"])
    assert mass == pytest.approx(1.0)
    assert dist == pytest.approx({"refund": 0.225, "rebooking": 0.675, "information": 0.1})


def test_label_that_is_a_prefix_of_another():
    positions = [
        pos("info", ("info", 1.0)),
        pos("<|im_end|>", ("<|im_end|>", 0.6), ("rmation", 0.4)),
    ]
    dist, _ = label_distribution(positions, ["info", "information"])
    assert dist == pytest.approx({"info": 0.6, "information": 0.4})


def test_leading_whitespace_token_is_skipped():
    positions = [pos("\n", ("\n", 0.9)), pos("no", ("no", 0.6), ("yes", 0.4))]
    dist, _ = label_distribution(positions, ["yes", "no"])
    assert dist["no"] == pytest.approx(0.6)


def test_off_script_answer_has_zero_mass():
    _, mass = label_distribution([pos("The", ("The", 0.9), ("I", 0.1))], ["yes", "no"])
    assert mass == 0


def test_ambiguous_prefix_at_end_of_tokens_is_split():
    dist, _ = label_distribution([pos("re", ("re", 1.0))], ["refund", "rebooking"])
    assert dist == pytest.approx({"refund": 0.5, "rebooking": 0.5})


def test_text_helpers():
    assert normalize(" Reply-Now.") == "replynow"
    assert label_from_text("Information, please", ["info", "information"]) == "information"
    assert label_from_text("<think>", ["yes", "no"]) is None
    assert one_hot(["a", "b"], "b") == {"a": 0.0, "b": 1.0}
    assert uniform(["a", "b", "c", "d"]) == {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
