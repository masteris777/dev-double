"""A thin synchronous client for the stunt-double server.

Keep this client behind your own interface in your codebase. When the real
decision engine is approved, write a second implementation of that interface
against the vendor's SDK and switch over; the rest of your harness stays.
"""

from __future__ import annotations

from typing import Any

import httpx


class StuntDouble:
    def __init__(self, base_url: str = "http://127.0.0.1:8787", timeout: float = 120.0) -> None:
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "StuntDouble":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        response = self._http.post(path, json={k: v for k, v in body.items() if v is not None})
        response.raise_for_status()
        return response.json()

    def decide(self, input: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return self._post("/v1/decide", {"input": input, "questions": questions})

    def route(self, input: Any, routes: dict[str, str] | None = None) -> dict[str, Any]:
        return self._post("/v1/route", {"input": input, "routes": routes})

    def guard(
        self,
        input: Any,
        policies: dict[str, str] | None = None,
        scope: str | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/v1/guard", {"input": input, "policies": policies, "scope": scope, "threshold": threshold}
        )

    def gate(
        self,
        tool: str,
        arguments: dict[str, Any] | str | None = None,
        context: Any = None,
        policy: str | None = None,
        outcomes: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/v1/gate",
            {
                "tool_call": {"name": tool, "arguments": arguments or {}},
                "context": context,
                "policy": policy,
                "outcomes": outcomes,
            },
        )

    def classify(
        self,
        labels: dict[str, str],
        input: Any = None,
        inputs: list[Any] | None = None,
        question: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/v1/classify", {"labels": labels, "input": input, "inputs": inputs, "question": question}
        )

    def judge(
        self,
        output: Any,
        input: Any = None,
        reference: Any = None,
        criteria: str | None = None,
        levels: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/v1/judge",
            {"output": output, "input": input, "reference": reference, "criteria": criteria, "levels": levels},
        )

    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None, return_documents: bool = False
    ) -> dict[str, Any]:
        return self._post(
            "/v1/rerank",
            {"query": query, "documents": documents, "top_n": top_n, "return_documents": return_documents},
        )

    def extract(self, input: Any, fields: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """``fields``: name -> {"type": string|number|integer|boolean|enum, "description"?, "options"?}."""
        return self._post("/v1/extract", {"input": input, "fields": fields})
