import pytest
from _shared import images

from dev_double.core.decision import systemone_wire as wire
from dev_double.core.decision.t_answer import TBinaryAnswer, TChoiceAnswer, TScaleAnswer
from dev_double.core.decision.t_question import TBinaryQuestion, TChoiceQuestion, TScaleQuestion
from dev_double.core.decision.t_usage import TUsage

CHOICE = {
    "type": "choice",
    "instructions": "Which label fits this ticket?",
    "criteria": {"billing": "Payments and refunds", "bug": "Software errors", "account": None},
}


def test_choice_from_wire_uses_the_key_for_a_null_description():
    q = wire.question_from_wire(CHOICE)
    assert q == TChoiceQuestion(
        "Which label fits this ticket?",
        {"billing": "Payments and refunds", "bug": "Software errors", "account": "account"},
    )


def test_noul_from_wire_maps_true_false_to_yes_no():
    q = wire.question_from_wire(
        {"type": "noul", "instructions": "Refund?", "criteria": {"true": "asks for money", "false": "does not"}}
    )
    assert q == TBinaryQuestion("Refund?", yes="asks for money", no="does not")
    assert wire.question_from_wire({"type": "noul", "instructions": "Refund?"}) == TBinaryQuestion("Refund?")
    only_false = wire.question_from_wire({"type": "noul", "instructions": "?", "criteria": {"false": "no"}})
    assert only_false == TBinaryQuestion("?", no="no")


def test_score_from_wire():
    q = wire.question_from_wire({"type": "score", "instructions": "How good?", "criteria": ["bad", "ok", "good"]})
    assert q == TScaleQuestion("How good?", ["bad", "ok", "good"])


def test_instructions_may_be_an_object_or_array_serialized_as_json():
    q = wire.question_from_wire({"type": "noul", "instructions": {"check": "refund", "n": 1}})
    assert q == TBinaryQuestion('{"check": "refund", "n": 1}')
    q = wire.question_from_wire({"type": "noul", "instructions": ["a", "é"]})
    assert q == TBinaryQuestion('["a", "é"]')


@pytest.mark.parametrize(
    "question",
    [
        TBinaryQuestion("Refund?"),
        TBinaryQuestion("Refund?", yes="asks for money"),
        TBinaryQuestion("Refund?", yes="asks for money", no="does not"),
        TChoiceQuestion("Which?", {"a": "A", "b": "B", "c": "C"}),
        TScaleQuestion("How?", ["low", "mid", "high"]),
    ],
)
def test_question_round_trips(question):
    assert wire.question_from_wire(wire.question_to_wire(question)) == question


def test_question_to_wire_shapes():
    assert wire.question_to_wire(TBinaryQuestion("Q?")) == {"type": "noul", "instructions": "Q?"}
    assert wire.question_to_wire(TBinaryQuestion("Q?", yes="y", no="n")) == {
        "type": "noul",
        "instructions": "Q?",
        "criteria": {"true": "y", "false": "n"},
    }
    assert wire.question_to_wire(TScaleQuestion("Q?", ["a", "b"])) == {
        "type": "score",
        "instructions": "Q?",
        "criteria": ["a", "b"],
    }


def test_the_contract_limits_are_accepted():
    many = {f"option{i}": f"d{i}" for i in range(26)}
    wire.question_from_wire({"type": "choice", "instructions": "?", "criteria": many})
    wire.question_from_wire({"type": "score", "instructions": "?", "criteria": [f"l{i}" for i in range(26)]})
    wire.questions_from_wire({f"q{i}": {"type": "noul", "instructions": "?"} for i in range(64)})


@pytest.mark.parametrize(
    "question, message",
    [
        ("text", "must be an object"),
        ({"type": "bad", "instructions": "?"}, "must be one of choice, noul, score"),
        ({"type": "noul"}, "instructions"),
        ({"type": "noul", "instructions": "  "}, "must not be blank"),
        ({"type": "noul", "instructions": 5}, "string, object, or array"),
        ({"type": "choice", "instructions": "?"}, "must be an object of option"),
        ({"type": "choice", "instructions": "?", "criteria": {"a": "only one"}}, "2 to 26 entries, got 1"),
        (
            {"type": "choice", "instructions": "?", "criteria": {f"o{i}": "x" for i in range(27)}},
            "2 to 26 entries, got 27",
        ),
        ({"type": "choice", "instructions": "?", "criteria": {"a": "x", " ": "y"}}, "keys must not be blank"),
        ({"type": "choice", "instructions": "?", "criteria": {"a": 1, "b": "y"}}, "must be a string or null"),
        ({"type": "choice", "instructions": "?", "criteria": {"yes": "x", "YES!": "y"}}, "too similar"),
        ({"type": "noul", "instructions": "?", "criteria": {"maybe": "x"}}, '"true" and "false"'),
        ({"type": "noul", "instructions": "?", "criteria": {"true": 1}}, "must be a string"),
        ({"type": "score", "instructions": "?", "criteria": ["one"]}, "2 to 26 entries, got 1"),
        ({"type": "score", "instructions": "?", "criteria": ["x"] * 27}, "2 to 26 entries, got 27"),
        ({"type": "score", "instructions": "?", "criteria": {"0": "a"}}, "list of strings"),
    ],
)
def test_question_from_wire_rejects_contract_violations(question, message):
    with pytest.raises(ValueError, match=message):
        wire.question_from_wire(question, "questions.q")


def test_error_messages_name_the_offending_path():
    with pytest.raises(ValueError, match=r"`questions\.label\.criteria`"):
        wire.questions_from_wire({"label": {"type": "choice", "instructions": "?", "criteria": {"a": "x"}}})


def test_questions_from_wire_validates_count_and_names():
    with pytest.raises(ValueError, match="1 to 64 entries, got 0"):
        wire.questions_from_wire({})
    with pytest.raises(ValueError, match="1 to 64 entries, got 65"):
        wire.questions_from_wire({f"q{i}": {"type": "noul", "instructions": "?"} for i in range(65)})
    with pytest.raises(ValueError, match="keys must not be blank"):
        wire.questions_from_wire({" ": {"type": "noul", "instructions": "?"}})
    with pytest.raises(ValueError, match="object"):
        wire.questions_from_wire([])


def test_state_from_wire():
    assert wire.state_from_wire("text") == "text"
    assert wire.state_from_wire({"a": 1}) == {"a": 1}
    assert wire.state_from_wire([1, 2]) == [1, 2]
    for bad in ("", "   ", None, 5):
        with pytest.raises(ValueError, match="state"):
            wire.state_from_wire(bad)


def test_answers_to_wire():
    assert wire.answer_to_wire(TBinaryAnswer(value=True, probability=0.83, confidence=0.66)) == {
        "type": "noul",
        "noul": 0.83,
    }
    assert wire.answer_to_wire(TChoiceAnswer("bug", {"bug": 0.9, "billing": 0.1}, 0.8)) == {
        "type": "choice",
        "choice": "bug",
        "probabilities": {"bug": 0.9, "billing": 0.1},
        "confidence": 0.531,  # recomputed as 1 - H(p) / ln(n), not taken from the answer
    }
    scale = TScaleAnswer(
        value=1.8,
        level=2,
        probabilities={"0": 0.1, "1": 0.1, "2": 0.8},
        legend={"0": "a", "1": "b", "2": "c"},
        confidence=0.7,
    )
    assert wire.answer_to_wire(scale) == {
        "type": "score",
        "score": 1.8,
        "legend": {"0": "a", "1": "b", "2": "c"},
        "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
        "confidence": 0.4183,
    }


def test_probabilities_from_wire():
    p = wire.probabilities_from_wire(TBinaryQuestion("?"), {"type": "noul", "noul": 0.25})
    assert p == {"yes": 0.25, "no": 0.75}
    q = TChoiceQuestion("?", {"a": "A", "b": "B"})
    assert wire.probabilities_from_wire(q, {"probabilities": {"b": 0.9, "a": 0.1}}) == {"a": 0.1, "b": 0.9}
    s = TScaleQuestion("?", ["lo", "hi"])
    assert wire.probabilities_from_wire(s, {"probabilities": {"0": 0.4, "1": 0.6}}) == {"0": 0.4, "1": 0.6}


AB = TChoiceQuestion("?", {"a": "A", "b": "B"})


@pytest.mark.parametrize(
    "question, answer, message",
    [
        (TBinaryQuestion("?"), {}, "noul"),
        (TBinaryQuestion("?"), {"noul": "high"}, "noul"),
        (TBinaryQuestion("?"), {"noul": True}, "noul"),
        (TBinaryQuestion("?"), {"noul": 1.5}, "noul"),
        (AB, {}, "probabilities"),
        (AB, {"probabilities": {"a": 1.0}}, "exactly the keys"),
        (AB, {"probabilities": {"a": 1.0, "b": 0.0, "c": 0.0}}, "exactly the keys"),
        (AB, {"probabilities": {"a": "x", "b": 0.0}}, "non-negative"),
        (TScaleQuestion("?", ["l", "h"]), {"probabilities": {"1": 0.5, "2": 0.5}}, "exactly the keys"),
        (TScaleQuestion("?", ["l", "h"]), "text", "must be an object"),
    ],
)
def test_probabilities_from_wire_rejects_bad_answers(question, answer, message):
    with pytest.raises(ValueError, match=message):
        wire.probabilities_from_wire(question, answer)


def test_request_and_response_bodies():
    questions = {"q": TBinaryQuestion("Refund?", yes="money")}
    assert wire.request_to_wire("nimble", {"t": 1}, questions) == {
        "model": "nimble",
        "state": {"t": 1},
        "questions": {"q": {"type": "noul", "instructions": "Refund?", "criteria": {"true": "money"}}},
    }
    answers = {"q": TBinaryAnswer(value=False, probability=0.1, confidence=0.8)}
    assert wire.response_to_wire("nimble", answers, TUsage(7, 1)) == {
        "model": "nimble",
        "answers": {"q": {"type": "noul", "noul": 0.1}},
        "usage": {"input_tokens": 7, "output_tokens": 1},
    }
    assert wire.usage_from_wire({"input_tokens": 3, "output_tokens": 2}) == TUsage(3, 2)
    assert wire.usage_from_wire(None) == TUsage(0, 0)


def test_wire_confidence_is_normalized_entropy_as_in_ollamas_example():
    # Ollama's published example: confidence 0.8906 (its probabilities are rounded to 4 places).
    probs = {"billing": 0.0125, "bug": 0.9781, "account": 0.0093}
    answer = TChoiceAnswer(value="bug", probabilities=probs, confidence=0.0)
    assert wire.answer_to_wire(answer)["confidence"] == pytest.approx(0.8906, abs=1e-3)


def test_request_to_wire_carries_images_in_order_only_when_there_are_some():
    questions = {"q": TBinaryQuestion("Refund?")}
    assert "images" not in wire.request_to_wire("nimble", "t", questions)
    assert "images" not in wire.request_to_wire("nimble", "t", questions, ())
    body = wire.request_to_wire("nimble", "t", questions, ("AAA=", "BBB="))
    assert body["images"] == ["AAA=", "BBB="]


def test_images_from_wire_accepts_png_jpeg_and_webp():
    assert wire.images_from_wire(None) == ()
    assert wire.images_from_wire([]) == ()
    assert wire.images_from_wire([images.PNG, images.JPEG, images.WEBP]) == (images.PNG, images.JPEG, images.WEBP)


@pytest.mark.parametrize(
    "bad, message",
    [
        ("data:image/png;base64," + images.PNG, "not a data URL"),
        ("https://example.com/cat.png", "not a URL"),
        ("file:///tmp/cat.png", "not a URL"),
        ("this is not base64!", "not valid base64"),
        (images.b64(b"hello world, definitely text"), "not a PNG, JPEG, or WebP"),
        (images.b64(b"GIF89a" + b"\x00" * 20), "not a PNG, JPEG, or WebP"),
        ("", "not a PNG, JPEG, or WebP"),
        (5, "must be a base64 string"),
    ],
)
def test_images_from_wire_rejects_what_is_not_plain_base64_of_a_supported_image(bad, message):
    with pytest.raises(ValueError, match=message) as exc:
        wire.images_from_wire([images.PNG, bad])
    assert "images[1]" in str(exc.value)


def test_images_from_wire_needs_a_list():
    with pytest.raises(ValueError, match="array"):
        wire.images_from_wire(images.PNG)
