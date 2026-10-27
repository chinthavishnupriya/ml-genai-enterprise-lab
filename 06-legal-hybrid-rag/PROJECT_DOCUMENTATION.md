# Project 06 — Legal Contract Q&A with Hybrid Search

## 1. Project overview

Search fictional contract documents and return relevant contract clauses with clause-level citations. The implementation combines keyword and lexical vector rankings, fuses their ranks, and reranks candidates. It returns source text for review rather than generating legal conclusions.

## 2. Technology stack

| Technology | Purpose |
|---|---|
| Python | Clause parser, retrieval pipeline, evaluator and CLI |
| Python standard library | Tokenization, counters, math and BM25-style scoring |
| Optional `sentence-transformers` | Dense sentence embeddings and cross-encoder reranking when separately installed |
| `unittest` | Ingestion, ranking, RRF, citations, abstention and regression tests |

The default workflow is offline and dependency-light. The default vector backend is a locally implemented TF-IDF lexical index. The optional dense backend downloads model files and may need more memory and disk.

## 3. Document ingestion

`load_contract()` parses Markdown/TXT files. Markdown headings act as clause boundaries; numeric clause IDs are separated from their heading titles when possible. A clause record keeps its identifier, title, source filename and text. Plain text without headings is indexed as a single fallback section. **PDF ingestion is not implemented by the current contract loader.**

## 4. Hybrid retrieval flow

1. Tokenize query and contract clause text; remove common stop words.
2. Retrieve rankings separately with BM25-style keyword relevance and TF-IDF cosine similarity.
3. Combine the two rankings with Reciprocal Rank Fusion (RRF).
4. Rerank candidates using query overlap with clause title/content.
5. Optionally use Sentence Transformers and a cross-encoder when explicitly configured.
6. Return top clause text, retrieval scores and clause citations. An empty match set causes abstention.

## 5. Source map

| File | Responsibility |
|---|---|
| `src/legal_rag.py` | Clause ingestion, BM25, TF-IDF, optional dense index, RRF and answer construction |
| `src/cli.py` | Query options and JSON response |
| `src/evaluate.py` | Six fixed query-to-clause regression cases, Recall@K and MRR@K |
| `data/contracts/` | Fictional SaaS agreement, data-processing addendum and SLA |
| `tests/test_legal_rag.py` | Ingestion, clause extraction, ranking, citation and abstention tests |
| `VALIDATION_REPORT.json` | Recorded Colab test/retrieval results |

## 6. Run locally

```bash
python -m unittest discover -s tests -v
python -m src.evaluate
python -m src.cli "When are subscription invoices due for payment?" --json
python -m src.cli "How quickly must a personal data breach be reported?"
```

Optional dense retrieval:

```bash
python -m pip install "sentence-transformers>=3,<6"
python -m src.cli "When are subscription invoices due for payment?" --vector-backend sentence-transformers --reranker cross-encoder --json
```

The first dense run may download embedding/reranking models. Do not install optional components unless you plan to run that path.

## 7. Validation

Sixteen tests and a six-query synthetic retrieval evaluation were recorded as passing. The evaluator reported Recall@5 = 1.0 and MRR@5 = 1.0 for its hand-labelled fictional contract corpus. These values describe the test cases only and are not legal correctness metrics.

## 8. Legal, privacy and quality limitations

- The included contract language is fictional and is not an authoritative statement of law.
- A citation proves only which local clause was retrieved; it does not prove validity, enforceability or applicability.
- Lexical search may miss paraphrases or relevant clauses outside top-K candidates.
- Do not ingest privileged or confidential material without proper authorization and security controls.
- Real use requires representative evaluation, citation verification and qualified legal review.
