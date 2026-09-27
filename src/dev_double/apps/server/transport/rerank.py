"""/v1/rerank: Cohere-style wire format, the de facto standard shared by
Hugging Face TEI, Infinity, Jina, and vLLM."""

from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, Field

from ....core.decision.t_rerank import TRerankRequest, TRerankResponse
from .common import Meta


class RerankDocument(BaseModel):
    text: str


class RerankRequest(BaseModel):
    query: str
    documents: list[Union[str, RerankDocument]] = Field(min_length=1)
    top_n: Optional[int] = Field(None, ge=1)
    return_documents: bool = False
    model: Optional[str] = Field(None, description="Ignored; the server's configured model is used.")

    def texts(self) -> list[str]:
        return [d if isinstance(d, str) else d.text for d in self.documents]

    def to_core(self) -> TRerankRequest:
        return TRerankRequest(
            query=self.query, documents=self.texts(), top_n=self.top_n, return_documents=self.return_documents
        )


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[RerankDocument] = None


class RerankResponse(BaseModel):
    id: str
    results: list[RerankResult]
    meta: Meta

    @classmethod
    def from_core(cls, t: TRerankResponse) -> "RerankResponse":
        results = [
            RerankResult(
                index=r.index,
                relevance_score=r.relevance_score,
                document=RerankDocument(text=r.document) if r.document is not None else None,
            )
            for r in t.results
        ]
        return cls(id=t.id, results=results, meta=Meta.from_core(t.meta))
