"""Hybrid retrieval over fictional contract clauses. Not legal advice."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[.'-][a-zA-Z0-9]+)*")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "for", "from", "how", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on",
    "or", "our", "please", "should", "that", "the", "their", "this", "to", "under", "was",
    "what", "when", "where", "which", "who", "will", "with", "would", "you", "your",
    "agreement", "contract", "fictional", "sample", "example", "demo", "document", "terms",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS]


@dataclass(frozen=True)
class Clause:
    clause_id: str
    title: str
    source: str
    text: str

    @property
    def searchable_text(self) -> str:
        return f"{self.clause_id} {self.title}\n{self.text}"

    @property
    def citation(self) -> str:
        return f"{self.source}, Clause {self.clause_id} — {self.title}"


@dataclass(frozen=True)
class ScoredClause:
    clause: Clause
    bm25_score: float
    vector_score: float
    rrf_score: float
    rerank_score: float

    @property
    def citation(self) -> str:
        return self.clause.citation


@dataclass(frozen=True)
class LegalAnswer:
    query: str
    answer: str
    citations: tuple[str, ...]
    retrieved: tuple[ScoredClause, ...]
    abstained: bool


def load_contract(path: str | Path) -> list[Clause]:
    """Load a Markdown/TXT contract with headings treated as clause boundaries."""
    path = Path(path)
    if path.suffix.lower() not in {".md", ".markdown", ".txt"}:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    if not path.is_file():
        raise FileNotFoundError(path)
    content = path.read_text(encoding="utf-8")
    clauses: list[Clause] = []
    current_title: str | None = None
    current_body: list[str] = []
    counter = 0

    def add_clause() -> None:
        nonlocal counter, current_title, current_body
        body = "\n".join(current_body).strip()
        if current_title and body:
            counter += 1
            match = re.match(r"^(\d+(?:\.\d+)*|[A-Z]{1,3}-\d+)\s*[.:\-—]?\s*(.*)$", current_title)
            if match:
                clause_id = match.group(1)
                title = match.group(2).strip(" .:-—") or current_title
            else:
                clause_id = f"S{counter:02d}"
                title = current_title.strip()
            clauses.append(Clause(clause_id=clause_id, title=title, source=path.name, text=body))
        current_body = []

    for line in content.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            add_clause()
            current_title = heading.group(1).strip()
        elif current_title is not None:
            current_body.append(line)
    add_clause()
    if not clauses:
        # Plain text without headings remains searchable as one document section.
        stripped = content.strip()
        if stripped:
            clauses = [Clause("S01", path.stem, path.name, stripped)]
    return clauses


def load_corpus(paths: Sequence[str | Path]) -> list[Clause]:
    clauses: list[Clause] = []
    for path in paths:
        clauses.extend(load_contract(path))
    if not clauses:
        raise ValueError("No non-empty clauses were loaded")
    return clauses


class BM25Index:
    """BM25Okapi-style keyword retrieval implemented with the standard library."""

    def __init__(self, clauses: Sequence[Clause], k1: float = 1.5, b: float = 0.75) -> None:
        if not clauses:
            raise ValueError("Index requires at least one clause")
        self.clauses = list(clauses)
        self.k1, self.b = k1, b
        self.counts = [Counter(tokenize(c.searchable_text)) for c in self.clauses]
        self.lengths = [sum(counts.values()) for counts in self.counts]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        df: Counter[str] = Counter()
        for counts in self.counts:
            df.update(counts.keys())
        n_docs = len(self.clauses)
        self.idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def rank(self, query: str) -> list[tuple[int, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        scores: list[tuple[int, float]] = []
        for i, counts in enumerate(self.counts):
            score = 0.0
            dl = self.lengths[i]
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
            if score > 0:
                scores.append((i, score))
        return sorted(scores, key=lambda item: (-item[1], self.clauses[item[0]].source, self.clauses[item[0]].clause_id))


class TfidfVectorIndex:
    """Local cosine retrieval over TF-IDF term vectors (sparse lexical vector fallback)."""

    def __init__(self, clauses: Sequence[Clause]) -> None:
        if not clauses:
            raise ValueError("Index requires at least one clause")
        self.clauses = list(clauses)
        self.term_counts = [Counter(tokenize(c.searchable_text)) for c in self.clauses]
        df: Counter[str] = Counter()
        for counts in self.term_counts:
            df.update(counts.keys())
        n_docs = len(self.clauses)
        self.idf = {term: math.log((n_docs + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}
        self.vectors = [self._vector(counts) for counts in self.term_counts]
        self.norms = [math.sqrt(sum(value * value for value in vector.values())) for vector in self.vectors]

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        return {
            term: (1.0 + math.log(tf)) * self.idf[term]
            for term, tf in counts.items() if term in self.idf
        }

    def rank(self, query: str) -> list[tuple[int, float]]:
        q_counts = Counter(tokenize(query))
        q_vector = self._vector(q_counts)
        q_norm = math.sqrt(sum(value * value for value in q_vector.values()))
        if q_norm == 0:
            return []
        scores: list[tuple[int, float]] = []
        for i, vector in enumerate(self.vectors):
            norm = self.norms[i]
            if norm == 0:
                continue
            dot = sum(weight * vector.get(term, 0.0) for term, weight in q_vector.items())
            score = dot / (q_norm * norm)
            if score > 0:
                scores.append((i, score))
        return sorted(scores, key=lambda item: (-item[1], self.clauses[item[0]].source, self.clauses[item[0]].clause_id))


class SentenceTransformerVectorIndex:
    """Optional dense sentence embedding index; install sentence-transformers to enable."""

    def __init__(self, clauses: Sequence[Clause], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "Dense embeddings require the optional sentence-transformers package. "
                "Install it separately; the default TF-IDF vector fallback works without it."
            ) from exc
        if not clauses:
            raise ValueError("Index requires at least one clause")
        self.clauses = list(clauses)
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [c.searchable_text for c in clauses],
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        self.np = np

    def rank(self, query: str) -> list[tuple[int, float]]:
        if not query.strip():
            return []
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.embeddings @ q
        ranked = [(i, float(score)) for i, score in enumerate(scores) if score > 0]
        return sorted(ranked, key=lambda item: (-item[1], self.clauses[item[0]].source, self.clauses[item[0]].clause_id))


class LegalHybridRAG:
    """BM25 + vector retrieval, Reciprocal Rank Fusion, and lightweight reranking."""

    def __init__(
        self,
        clauses: Sequence[Clause],
        *,
        vector_backend: str = "tfidf",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rrf_k: int = 60,
    ) -> None:
        if not clauses:
            raise ValueError("Cannot create a legal index without clauses")
        if vector_backend not in {"tfidf", "sentence-transformers"}:
            raise ValueError("vector_backend must be 'tfidf' or 'sentence-transformers'")
        self.clauses = list(clauses)
        self.bm25 = BM25Index(self.clauses)
        if vector_backend == "sentence-transformers":
            self.vector_index = SentenceTransformerVectorIndex(self.clauses, embedding_model)
        else:
            self.vector_index = TfidfVectorIndex(self.clauses)
        self.vector_backend = vector_backend
        self.rrf_k = rrf_k

    @classmethod
    def from_paths(cls, paths: Sequence[str | Path], **kwargs) -> "LegalHybridRAG":
        return cls(load_corpus(paths), **kwargs)

    @staticmethod
    def _rank_dict(ranked: list[tuple[int, float]]) -> dict[int, tuple[int, float]]:
        return {doc_index: (rank, score) for rank, (doc_index, score) in enumerate(ranked, start=1)}

    def retrieve(self, query: str, *, top_k: int = 5, candidate_k: int = 12) -> list[ScoredClause]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if top_k < 1 or candidate_k < 1:
            raise ValueError("top_k and candidate_k must be positive")
        bm_rank = self._rank_dict(self.bm25.rank(query)[:candidate_k])
        vec_rank = self._rank_dict(self.vector_index.rank(query)[:candidate_k])
        candidate_ids = set(bm_rank) | set(vec_rank)
        query_terms = set(tokenize(query))
        results: list[ScoredClause] = []
        for index in candidate_ids:
            bm_r, bm_score = bm_rank.get(index, (None, 0.0))
            vec_r, vec_score = vec_rank.get(index, (None, 0.0))
            rrf = (1.0 / (self.rrf_k + bm_r) if bm_r else 0.0) + (1.0 / (self.rrf_k + vec_r) if vec_r else 0.0)
            clause = self.clauses[index]
            title_overlap = len(query_terms & set(tokenize(clause.title)))
            content_terms = set(tokenize(clause.text))
            content_overlap = len(query_terms & content_terms) / max(1, len(query_terms))
            rerank_score = rrf + 0.002 * title_overlap + 0.001 * content_overlap
            results.append(ScoredClause(clause, bm_score, vec_score, rrf, rerank_score))
        results.sort(key=lambda r: (-r.rerank_score, -r.rrf_score, r.clause.source, r.clause.clause_id))
        return results[:top_k]

    def answer(self, query: str, *, top_k: int = 3, max_clauses: int = 2) -> LegalAnswer:
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return LegalAnswer(
                query,
                "I could not find a relevant clause in the indexed fictional agreements. "
                "This system cannot determine your legal rights or provide legal advice; consult a qualified lawyer for advice on your situation.",
                (), (), True,
            )
        chosen = results[:max(1, max_clauses)]
        parts = [
            f"- **{r.clause.title}**: {r.clause.text.strip()} [{r.citation}]"
            for r in chosen
        ]
        answer = (
            "Relevant text from the indexed contract documents (informational retrieval only; not legal advice):\n"
            + "\n".join(parts)
        )
        return LegalAnswer(
            query=query,
            answer=answer,
            citations=tuple(r.citation for r in chosen),
            retrieved=tuple(results),
            abstained=False,
        )
