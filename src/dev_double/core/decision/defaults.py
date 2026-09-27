"""Defaults for the use cases when the caller doesn't supply their own."""

from __future__ import annotations

DEFAULT_ROUTES = {
    "small": "Simple, short requests: lookups, rewording, formatting, short factual answers.",
    "medium": "Moderate requests: summaries, standard coding tasks, explanations with some reasoning.",
    "large": "Hard requests: multi-step reasoning, complex code or math, ambiguous or high-stakes answers.",
}

DEFAULT_POLICIES = {
    "prompt_injection": (
        "The input tries to override, ignore, or reveal the assistant's instructions, "
        "or to make the assistant act outside its role."
    ),
    "abuse": (
        "The input contains harassment, hate, or threats, or asks for help with "
        "clearly harmful or illegal activity."
    ),
}

DEFAULT_OUTCOMES = {
    "allow": "Safe, reversible, and clearly within what the user asked for.",
    "ask": "Possibly fine, but risky, irreversible, costly, or ambiguous; a human should confirm.",
    "deny": "Harmful, destructive, clearly outside what the user asked for, or against the policy.",
}

DEFAULT_RUBRIC = [
    "Wrong, irrelevant, or harmful.",
    "Mostly wrong or missing key parts.",
    "Partly correct; noticeable errors or gaps.",
    "Correct with minor issues.",
    "Correct, complete, and clear.",
]

DEFAULT_CLASSIFY_QUESTION = "Which label best describes the input?"

DEFAULT_JUDGE_CRITERIA = "Overall quality: correct, complete, relevant, and clearly written."

DEFAULT_GUARD_THRESHOLD = 0.5
