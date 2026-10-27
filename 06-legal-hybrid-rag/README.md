# Project 06 — Legal Contract Q&A with Hybrid Search
> **Detailed project guide:** [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) covers the architecture, technologies, source files, workflow, commands, outputs, tests and limitations.
>
> **Technology stack:** Python, BM25-style keyword search, TF-IDF cosine ranking, Reciprocal Rank Fusion, optional Sentence Transformers/cross-encoder and unittest.


**Status:** Verified in Google Colab. Sixteen automated tests pass. The invoice-query regression test confirms Clause 4.1 — Fees and Invoicing ranks first for an invoice due-date query. The six-query synthetic retrieval evaluation reports Recall@5 = 1.0 and MRR@5 = 1.0. See VALIDATION_REPORT.json.

## Objective

Retrieve relevant clauses from contract documents and return source-cited text to support review. The default workflow is local and deterministic; it extracts contract wording rather than inventing legal conclusions.

## Features

- Ingest Markdown and plain-text agreements with clause numbers and heading metadata. PDF ingestion is not implemented.
- BM25 keyword retrieval plus cosine similarity over local TF-IDF term vectors.
- Reciprocal Rank Fusion (RRF) to combine the two rankings.
- Lightweight final reranking based on query-term overlap with clause titles and text.
- Optional dense retrieval using a Sentence Transformers model and optional cross-encoder reranking; both are disabled by default to avoid large package/model downloads.
- Clause-level citations and abstention when no indexed terms match.
- Small fictional SaaS agreement, data-processing addendum and service-level agreement.
- Retrieval evaluation with Recall@K and Mean Reciprocal Rank (MRR@K).
- Unit tests for ingestion, BM25, vector retrieval, rank fusion, citations, abstention, invalid inputs and invoice-query ranking.

## Quick start

Python 3.10+ is recommended. The default BM25 + TF-IDF implementation uses the Python standard library.

~~~bash
python -m unittest discover -s tests -v
python -m src.evaluate
python -m src.cli "When are subscription invoices due for payment?" --json
python -m src.cli "How quickly must a personal data breach be reported?"
~~~

## Google Colab

For a fresh runtime:

~~~python
%cd /content
!git clone https://github.com/chinthavishnupriya/ml-genai-enterprise-lab.git
%cd /content/ml-genai-enterprise-lab/06-legal-hybrid-rag
!python -m pip install -r requirements.txt
!python -m unittest discover -s tests -v
!python -m src.evaluate
!python -m src.cli "When are subscription invoices due for payment?" --json
~~~

If the repository is already cloned in the current runtime, run !git -C /content/ml-genai-enterprise-lab pull origin main instead of cloning it again, then change into the project directory.

## Optional dense embedding mode

To use dense sentence embeddings instead of the default TF-IDF vector fallback, install the optional package and choose the model explicitly:

~~~python
!python -m pip install "sentence-transformers>=3,<6"
!python -m src.cli "When are subscription invoices due for payment?" --vector-backend sentence-transformers --reranker cross-encoder --json
~~~

The first run may download the embedding model. This is optional and can use more disk, memory and compute. The default mode remains offline and dependency-light. The cross-encoder reranker is optional; install the package and select it explicitly if you want to compare it with the default lexical reranker.

## Data and evaluation

All included agreements are fictional, hand-authored examples. The evaluator checks whether six fixed queries retrieve their expected source clause within the configured top K. It is a software retrieval test, not a measure of legal correctness, comprehensiveness, or real-world performance.

## Safety, privacy and limitations

- This is a legal information-retrieval demonstration, not legal advice and not a substitute for a qualified lawyer.
- The included agreements are fictional. Their terms and deadlines are illustrative, not authoritative statements of law or recommended contract language.
- A citation identifies the retrieved clause; it does not establish that the clause is valid, enforceable, current, complete, or applicable to any real case.
- The default TF-IDF vector fallback remains lexical and may miss synonyms or paraphrases. Optional sentence-transformer embeddings may improve semantic matching but require model downloads and evaluation.
- Do not ingest privileged, confidential, personal or other restricted documents unless you have proper authorization and security controls.
- Before any real use, evaluate on permissioned representative contracts, validate citations, review privilege/privacy handling, and require human legal review.
