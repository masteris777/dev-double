"""Token usage, accumulated across the model calls of one request."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "TUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
