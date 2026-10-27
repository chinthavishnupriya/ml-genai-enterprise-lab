"""Typed domain objects for the enterprise RAG prototype."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    section: str
    text: str
    chunk_id: str

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.section}"


@dataclass(frozen=True)
class SearchHit:
    chunk: KnowledgeChunk
    score: float
    bm25_score: float
    tfidf_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.chunk.source,
            "section": self.chunk.section,
            "citation": self.chunk.citation,
            "score": round(self.score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "tfidf_score": round(self.tfidf_score, 4),
            "text": self.chunk.text,
        }


@dataclass(frozen=True)
class ToolResult:
    tool: str
    success: bool
    summary: str
    data: dict[str, Any] | None = None
    source: str | None = None


@dataclass(frozen=True)
class AgentResponse:
    question: str
    answer: str
    route: str
    citations: tuple[str, ...]
    abstained: bool
    trace: tuple[dict[str, Any], ...]
    safety: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
