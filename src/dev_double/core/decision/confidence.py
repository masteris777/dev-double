"""How concentrated a distribution is, and output rounding."""

from __future__ import annotations

import math

DIGITS = 4


def confidence(probabilities: list[float]) -> float:
    """How concentrated a distribution is: 1.0 when all mass is on one label,
    0.0 when it is spread evenly. Normalized max probability: (n*max - 1) / (n - 1).
    """
    n = len(probabilities)
    if n < 2:
        return 1.0
    return max(0.0, (n * max(probabilities) - 1) / (n - 1))


def entropy_confidence(probabilities: list[float]) -> float:
    """The System One wire's confidence: 1 - H(p) / ln(n). Same endpoints as
    ``confidence`` (1.0 all on one label, 0.0 uniform), different in between.
    """
    n = len(probabilities)
    total = sum(probabilities)
    if n < 2 or total <= 0:
        return 1.0
    entropy = -sum(p / total * math.log(p / total) for p in probabilities if p > 0)
    return min(1.0, max(0.0, 1 - entropy / math.log(n)))


def round_probabilities(probs: dict[str, float]) -> dict[str, float]:
    return {k: round(v, DIGITS) for k, v in probs.items()}
