"""Hybrid local retrieval combining BM25-style lexical ranking and TF-IDF cosine."""
from __future__ import annotations
from collections import Counter
import math
import re
from .models import KnowledgeChunk, SearchHit

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "is", "are", "was", "were",
    "be", "for", "in", "on", "with", "what", "when", "where", "how", "does",
    "do", "can", "i", "we", "you", "it", "this", "that", "my", "our", "your",
    "about", "please", "tell", "me", "show", "give", "from", "by", "as", "at",
}
# Normalize frequent inflections that matter for section-heading retrieval.
TOKEN_ALIASES = {
    "delays": "delay",
    "delayed": "delay",
    "delaying": "delay",
    "escalate": "escalation",
    "escalated": "escalation",
    "escalating": "escalation",
    "escalations": "escalation",
    "deliver": "delivery",
    "delivered": "delivery",
    "deliveries": "delivery",
}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.casefold())
    result = []
    for word in words:
        if word in STOPWORDS or len(word) <= 1:
            continue
        normalized = TOKEN_ALIASES.get(word, word)
        if normalized == word and len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            normalized = word[:-1]
        result.append(normalized)
    return result


class HybridRetriever:
    def __init__(self, chunks: list[KnowledgeChunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("At least one knowledge chunk is required")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(chunk.section + " " + chunk.text) for chunk in chunks]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        self.term_freqs = [Counter(tokens) for tokens in self.tokens]
        self.doc_freqs: Counter[str] = Counter()
        for terms in self.term_freqs:
            self.doc_freqs.update(terms.keys())
        self.idf = {
            term: math.log(1 + (len(chunks) - freq + 0.5) / (freq + 0.5))
            for term, freq in self.doc_freqs.items()
        }
        self.document_tfidf = [self._tfidf_vector(tokens) for tokens in self.tokens]

    def _tfidf_vector(self, tokens: list[str]) -> dict[str, float]:
        frequencies = Counter(tokens)
        vector = {
            term: (1 + math.log(count)) * self.idf.get(term, 0.0)
            for term, count in frequencies.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return {term: value / norm for term, value in vector.items()} if norm else {}

    def _bm25(self, query_terms: list[str], index: int) -> float:
        score = 0.0
        length = self.lengths[index]
        for term in query_terms:
            frequency = self.term_freqs[index].get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (1 - self.b + self.b * length / max(self.avgdl, 1))
            score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
        return score

    def _cosine(self, query_terms: list[str], index: int) -> float:
        q_freq = Counter(query_terms)
        q_vec = {
            term: (1 + math.log(count)) * self.idf.get(term, 0.0)
            for term, count in q_freq.items()
        }
        norm = math.sqrt(sum(value * value for value in q_vec.values()))
        if not norm:
            return 0.0
        q_vec = {term: value / norm for term, value in q_vec.items()}
        return sum(q_vec.get(term, 0.0) * value for term, value in self.document_tfidf[index].items())

    @staticmethod
    def _scale(values: list[float]) -> list[float]:
        maximum = max(values, default=0.0)
        if maximum <= 0:
            return [0.0 for _ in values]
        return [value / maximum for value in values]

    def search(self, query: str, *, top_k: int = 3) -> list[SearchHit]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must not be blank")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        q_terms = tokenize(query)
        if not q_terms:
            return []
        bm25 = [self._bm25(q_terms, i) for i in range(len(self.chunks))]
        tfidf = [self._cosine(q_terms, i) for i in range(len(self.chunks))]
        bm25_scaled = self._scale(bm25)
        tfidf_scaled = self._scale(tfidf)
        q_phrase = " ".join(q_terms)
        query_term_set = set(q_terms)
        scored = []
        for i, chunk in enumerate(self.chunks):
            if bm25[i] <= 0 and tfidf[i] <= 0:
                continue
            doc_terms = " ".join(self.tokens[i])
            phrase_bonus = 0.05 if q_phrase and q_phrase in doc_terms else 0.0
            section_terms = set(tokenize(chunk.section))
            # Prefer a specific heading that matches the user's normalized intent.
            heading_overlap = len(query_term_set & section_terms)
            heading_bonus = 0.08 * heading_overlap
            score = 0.65 * bm25_scaled[i] + 0.35 * tfidf_scaled[i] + phrase_bonus + heading_bonus
            scored.append(SearchHit(chunk, score, bm25_scaled[i], tfidf_scaled[i]))
        scored.sort(key=lambda hit: (-hit.score, hit.chunk.source, hit.chunk.section))
        return scored[:top_k]
