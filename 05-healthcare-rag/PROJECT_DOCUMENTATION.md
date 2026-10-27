# Project 05 — Healthcare Document Ingestion & Retrieval-Augmented Answers

## 1. Project overview

This project indexes local educational healthcare documents and returns short extractive answers with citations. If relevant evidence is not found, it abstains rather than inventing a clinical answer. It is a local retrieval demonstration, not diagnosis, triage or treatment software.

## 2. Technology stack

| Technology | Purpose |
|---|---|
| Python | Ingestion, chunking, retrieval, answer construction and CLI |
| Python standard library | Regular expressions, counters, math, dataclasses and path handling |
| `pypdf` (optional runtime dependency) | Extracting text from PDF files |
| `unittest` | Chunking, retrieval, citation and abstention tests |

The default lexical index is implemented locally rather than relying on an external vector database or a hosted LLM.

## 3. Ingestion and chunking

`load_source_pages()` supports Markdown, text and PDF files. PDF ingestion requires `pypdf` to be installed. Markdown heading-based chunking stores the heading as section metadata; fixed-size chunking supports overlapping windows. Chunks contain a stable local chunk identifier, source filename and optional page/section metadata for citations.

## 4. Retrieval and answer construction

`LocalTfidfIndex` tokenizes content, computes term/document frequencies and inverse-document-frequency-style weights, then ranks chunks using cosine similarity. A score threshold controls whether there is enough retrieval evidence to answer.

For an answered query, the system ranks sentences by query-term overlap combined with chunk relevance and returns a short extractive response with source/chunk citations. If relevant evidence is below threshold, it returns an explicit abstention and directs the user to a qualified healthcare professional for personal guidance.

This is lexical retrieval: it can miss synonyms or semantically related terms that do not share words with a query.

## 5. Source map

| File | Responsibility |
|---|---|
| `src/health_rag.py` | File loading, page/chunk models, lexical index, sentence ranking, citations and abstention |
| `src/cli.py` | Local query CLI with text or JSON output |
| `src/evaluate.py` | Three hand-authored retrieval evaluation queries |
| `data/knowledge_base/` | Synthetic appointment, medication and urgent-care notes |
| `tests/test_health_rag.py` | Chunking, retrieval ranking, citations, invalid input and abstention tests |
| `VALIDATION_REPORT.json` | Previously recorded test and retrieval smoke results |

## 6. Run locally

```bash
python -m venv .venv
# Activate .venv first
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.evaluate
python -m src.cli "What information should I prepare before an appointment?" --json
```

For PDF input, ensure `pypdf` is installed in that environment. The CLI searches supported files under `data/knowledge_base` by default; use `--data-dir` for a different local folder.

## 7. Validation and interpretation

Fourteen tests were recorded as passing, along with a three-query retrieval smoke evaluation. The three expected sources ranked within the tested top results in the recorded run. That is a small code-path/retrieval check, not a clinical validation study. PDF ingestion was not exercised in the recorded validation.

## 8. Limitations and responsible use

- The built-in corpus is educational and synthetic, not an authoritative clinical reference.
- Retrieval can fail on synonyms, miss context or extract a sentence that is incomplete when read alone.
- Citation provenance shows the retrieved source; it does not guarantee medical correctness.
- Personal medical decisions, emergency triage and treatment decisions belong with qualified healthcare professionals and appropriate emergency services.
- Before any health-related deployment, use approved sources, clinical governance, privacy controls, representative clinical evaluation and expert review.
