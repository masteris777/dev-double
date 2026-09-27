"""The content a question is asked about."""

from __future__ import annotations

from typing import Any, Union

# Text, an object, or a list. Objects and lists are shown to the model as JSON.
TInputValue = Union[str, dict[str, Any], list[Any]]
