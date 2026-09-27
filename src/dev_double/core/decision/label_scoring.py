"""Pure helpers that turn a model's token probabilities (or text) into a
distribution over a fixed set of labels. Shared by logprob-reading engines."""

from __future__ import annotations

import math
from typing import Optional, Sequence

from .t_label_query import TPosition


def normalize(text: str) -> str:
    """Lowercase and keep only letters and digits, so "Reply_now", " reply now"
    and "reply-now." all compare equal."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def label_distribution(
    positions: Sequence[TPosition], labels: list[str]
) -> tuple[dict[str, float], float]:
    """Turn per-token probabilities into a distribution over labels.

    Labels can span several tokens ("rebooking" may be "re" + "booking"). We
    walk the generated tokens: probability on a token that fits exactly one
    label goes to that label; if the generated token still fits several
    labels, we continue to the next position, where the model's conditional
    probabilities split it. Alternatives that fit several labels but were not
    generated are split evenly (we can't see how the model would continue them).

    Returns (distribution, label_mass): label_mass is how much of the model's
    probability landed on valid labels.
    """
    keys = {label: normalize(label) for label in labels}
    acc = dict.fromkeys(labels, 0.0)
    prefix, weight, started = "", 1.0, False

    for pos in positions:
        if not started and not normalize(pos.token):
            continue  # leading whitespace or punctuation
        started = True
        complete = [label for label, key in keys.items() if key == prefix and prefix]
        next_step: Optional[tuple[str, float]] = None
        for token, logprob in pos.candidates or [(pos.token, 0.0)]:
            p = weight * math.exp(logprob)
            extended = prefix + normalize(token)
            fits = [label for label, key in keys.items() if key.startswith(extended)]
            if extended != prefix:
                fits = [label for label in fits if keys[label] != prefix]
            if not fits or extended == prefix:
                # The token ends the answer (or is off-script): credit a label
                # that is already complete, if any.
                for label in complete:
                    acc[label] += p / len(complete)
            elif len(fits) == 1:
                acc[fits[0]] += p
            elif token == pos.token:
                next_step = (extended, p)
            else:
                for label in fits:
                    acc[label] += p / len(fits)
        if next_step is None:
            break
        prefix, weight = next_step
    else:
        # Ran out of tokens while the answer was still ambiguous.
        if started and prefix:
            fits = [label for label, key in keys.items() if key.startswith(prefix)]
            for label in fits:
                acc[label] += weight / len(fits)

    mass = sum(acc.values())
    if mass <= 0:
        return acc, 0.0
    return {k: v / mass for k, v in acc.items()}, mass


def label_from_text(text: str, labels: list[str]) -> Optional[str]:
    """The longest label the text starts with, ignoring case and punctuation."""
    norm = normalize(text)
    fits = [label for label in labels if normalize(label) and norm.startswith(normalize(label))]
    return max(fits, key=lambda label: len(normalize(label)), default=None)


def one_hot(labels: list[str], chosen: str) -> dict[str, float]:
    return {label: 1.0 if label == chosen else 0.0 for label in labels}


def uniform(labels: list[str]) -> dict[str, float]:
    return {label: 1.0 / len(labels) for label in labels}
