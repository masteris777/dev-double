"""The three question types every use case is built from.

- ``binary``: is this statement true? -> probability of "yes"
- ``choice``: which of these unordered options? -> option + distribution
- ``scale``:  which of these ordered levels? -> expected level + distribution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

# The System One contract allows 2 to 26 options or levels. An engine that can't
# score that many refuses the question (see ``UnsupportedRequestError``): the
# openai engine reads at most 20 candidate tokens, so it stops at 20.
MAX_OPTIONS = 26
MAX_LEVELS = 26


def check_option_keys(options: dict[str, str]) -> None:
    """The model answers with an option key, compared ignoring case and
    punctuation, so keys must stay distinct under that comparison."""
    seen: dict[str, str] = {}
    for key in options:
        norm = "".join(ch for ch in key.lower() if ch.isalnum())
        if not norm:
            raise ValueError(f"Option key {key!r} must contain a letter or digit.")
        if norm in seen:
            raise ValueError(f"Option keys {seen[norm]!r} and {key!r} are too similar; rename one.")
        seen[norm] = key


def check_options(options: dict[str, str], what: str = "options") -> None:
    """2 to MAX_OPTIONS options with distinct keys."""
    if not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"`{what}` must have 2 to {MAX_OPTIONS} entries, got {len(options)}.")
    check_option_keys(options)


def check_levels(levels: list[str], what: str = "levels") -> None:
    """2 to MAX_LEVELS ordered levels."""
    if not 2 <= len(levels) <= MAX_LEVELS:
        raise ValueError(f"`{what}` must have 2 to {MAX_LEVELS} entries, got {len(levels)}.")


@dataclass(frozen=True)
class TBinaryQuestion:
    question: str
    yes: Optional[str] = None  # what counts as yes
    no: Optional[str] = None  # what counts as no


@dataclass(frozen=True)
class TChoiceQuestion:
    question: str
    options: dict[str, str]  # option key -> description; keys are returned as the answer

    def __post_init__(self) -> None:
        check_options(self.options)


@dataclass(frozen=True)
class TScaleQuestion:
    question: str
    levels: list[str]  # ordered descriptions, lowest first; level i is returned as i

    def __post_init__(self) -> None:
        check_levels(self.levels)


TQuestion = Union[TBinaryQuestion, TChoiceQuestion, TScaleQuestion]
