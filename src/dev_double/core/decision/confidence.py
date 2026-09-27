"""How concentrated a distribution is, and output rounding."""

from __future__ import annotations

DIGITS = 4


def confidence(probabilities: list[float]) -> float:
    """How concentrated a distribution is: 1.0 when all mass is on one label,
    0.0 when it is spread evenly. Normalized max probability: (n*max - 1) / (n - 1).
    """
    n = len(probabilities)
    if n < 2:
        return 1.0
    return max(0.0, (n * max(probabilities) - 1) / (n - 1))


def round_probabilities(probs: dict[str, float]) -> dict[str, float]:
    return {k: round(v, DIGITS) for k, v in probs.items()}
