"""Local TF-IDF-style retrieval and extractive, citation-backed answers.

Educational demonstration only. This is not a diagnostic or clinical decision tool.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:['-][a-zA-Z0-9]+)*")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "for", "from", "how", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or",
    "please", "should", "tell", "that", "the", "their", "this", "to", "use", "what", "when",
    "where", "which", "who", "with", "would", "you", "your", "about", "after", "before", "during",
    "information", "fictional", "document", "included", "demonstrate", "retrieval", "software",
    "designed", "test", "educational", "only",
}


@dataclass(frozen=True)
class SourcePage:
    source: str
    text: str
    page: int | None = None


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    text: str
    section: str | None = None
    page: int | None = None

    @property
    def citation(self) -> str:
        location = f", p. {self.page}" if self.page is not None else ""
        section = f", section: {self.section}" if self.section else ""
        return f"{self.source}{location}{section} (chunk {self.chunk_id})"


@dataclass(frozen=True)
class Retrieval:
    chunk: DocumentChunk
    score: float


@dataclass(frozen=True)
class RAGAnswer:
    query: str
    answer: str
    citations: tuple[str, ...]
    retrieved: tuple[Retrieval, ...]
    abstained: bool


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS]


def load_source_pages(path: str | Path) -> list[SourcePage]:
    """Read Markdown/text and PDF. PDF support requires pypdf installed."""
    path = Path(path)
    suffix = path.suffix.casefold()
    if suffix in {".md", ".markdown", ".txt"}:
        return [SourcePage(path.name, path.read_text(encoding="utf-8"))]
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF ingestion requires pypdf. Install requirements.txt first.") from exc
        reader = PdfReader(str(path))
        return [SourcePage(path.name, page.extract_text() or "", i + 1) for i, page in enumerate(reader.pages)]
    raise ValueError(f"Unsupported document type: {suffix or '<no extension>'}")


def _markdown_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, current_lines))
    return [(heading, "\n".join(lines).strip()) for heading, lines in sections if "\n".join(lines).strip()]


def _fixed_windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if chunk_size < 80:
        raise ValueError("chunk_size must be at least 80 characters")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    output: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + int(chunk_size * 0.6), end)
            if boundary > start:
                end = boundary
        part = text[start:end].strip()
        if part:
            output.append(part)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return output


def chunk_sources(
    pages: Iterable[SourcePage], *, strategy: str = "heading", chunk_size: int = 700, overlap: int = 100
) -> list[DocumentChunk]:
    """Create stable chunks using Markdown headings or fixed-size overlap."""
    if strategy not in {"heading", "fixed"}:
        raise ValueError("strategy must be 'heading' or 'fixed'")
    chunks: list[DocumentChunk] = []
    for page in pages:
        if strategy == "heading" and page.source.lower().endswith((".md", ".markdown")):
            sections = _markdown_sections(page.text)
        else:
            sections = [(None, page.text)]
        for section, text in sections:
            for part in _fixed_windows(text, chunk_size, overlap):
                chunks.append(DocumentChunk(
                    chunk_id=f"C{len(chunks) + 1:04d}", source=page.source,
                    text=part, section=section, page=page.page,
                ))
    return chunks


class LocalTfidfIndex:
    """Small dependency-light sparse TF-IDF vector index with cosine ranking."""

    def __init__(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index without non-empty document chunks")
        self.chunks = list(chunks)
        self._term_counts = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self._document_frequency: Counter[str] = Counter()
        for counts in self._term_counts:
            self._document_frequency.update(counts.keys())
        n_docs = len(self.chunks)
        self._idf = {
            term: math.log(1.0 + (n_docs + 1.0) / (df + 1.0)) + 1.0
            for term, df in self._document_frequency.items()
        }
        self._vectors = [self._weighted_vector(counts) for counts in self._term_counts]
        self._norms = [math.sqrt(sum(value * value for value in vector.values())) for vector in self._vectors]

    def _weighted_vector(self, counts: Counter[str]) -> dict[str, float]:
        return {term: (1.0 + math.log(count)) * self._idf[term] for term, count in counts.items() if term in self._idf}

    def search(self, query: str, *, top_k: int = 4) -> list[Retrieval]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        q_counts = Counter(tokenize(query))
        q_vector = self._weighted_vector(q_counts)
        q_norm = math.sqrt(sum(value * value for value in q_vector.values()))
        if q_norm == 0:
            return []
        results: list[Retrieval] = []
        for chunk, vector, norm in zip(self.chunks, self._vectors, self._norms):
            if norm == 0:
                score = 0.0
            else:
                dot = sum(weight * vector.get(term, 0.0) for term, weight in q_vector.items())
                score = dot / (q_norm * norm)
            results.append(Retrieval(chunk, float(score)))
        results.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return results[:top_k]


def _candidate_sentences(retrieval: Retrieval, query_terms: set[str]) -> list[tuple[float, str, Retrieval]]:
    sentences = [s.strip() for s in SENTENCE_RE.split(retrieval.chunk.text) if s.strip()]
    ranked: list[tuple[float, str, Retrieval]] = []
    for sentence in sentences:
        terms = set(tokenize(sentence))
        overlap = len(query_terms & terms)
        if overlap:
            match_score = overlap / math.sqrt(max(1, len(terms)))
            ranked.append((retrieval.score * (1.0 + match_score), sentence, retrieval))
    return ranked


class HealthcareRAG:
    """Index local educational documents and return extractive answers with citations."""

    def __init__(self, chunks: Sequence[DocumentChunk], *, min_score: float = 0.04) -> None:
        self.index = LocalTfidfIndex(chunks)
        self.min_score = min_score

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path], *, strategy: str = "heading", chunk_size: int = 700, overlap: int = 100, min_score: float = 0.04) -> "HealthcareRAG":
        pages: list[SourcePage] = []
        for path in paths:
            pages.extend(load_source_pages(path))
        chunks = chunk_sources(pages, strategy=strategy, chunk_size=chunk_size, overlap=overlap)
        return cls(chunks, min_score=min_score)

    def retrieve(self, query: str, *, top_k: int = 4) -> list[Retrieval]:
        return self.index.search(query, top_k=top_k)

    def answer(self, query: str, *, top_k: int = 4, max_sentences: int = 3) -> RAGAnswer:
        candidates = self.retrieve(query, top_k=top_k)
        relevant = [item for item in candidates if item.score >= self.min_score]
        if not relevant:
            return RAGAnswer(
                query=query,
                answer=("I could not find sufficiently relevant evidence in the local knowledge base. "
                        "Do not use this prototype to diagnose, choose treatment, or delay urgent care. "
                        "For personal medical guidance, consult a qualified healthcare professional."),
                citations=(), retrieved=tuple(candidates), abstained=True,
            )
        query_terms = set(tokenize(query))
        sentence_candidates: list[tuple[float, str, Retrieval]] = []
        for item in relevant:
            sentence_candidates.extend(_candidate_sentences(item, query_terms))
        sentence_candidates.sort(key=lambda value: (-value[0], value[2].chunk.chunk_id, value[1]))
        chosen: list[tuple[str, Retrieval]] = []
        seen: set[str] = set()
        for _, sentence, retrieval in sentence_candidates:
            normalized = sentence.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            chosen.append((sentence, retrieval))
            if len(chosen) >= max_sentences:
                break
        if not chosen:
            best = relevant[0]
            chosen = [(best.chunk.text, best)]
        answer_lines = [f"- {sentence} [{retrieval.chunk.citation}]" for sentence, retrieval in chosen]
        citations = tuple(dict.fromkeys(retrieval.chunk.citation for _, retrieval in chosen))
        answer = "Evidence from the local educational documents (not medical advice):\n" + "\n".join(answer_lines)
        return RAGAnswer(query, answer, citations, tuple(candidates), False)
