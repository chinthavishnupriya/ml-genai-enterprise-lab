# Project 05 — Healthcare Clinical Manual Ingestion & RAG
> **Detailed project guide:** [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) covers the architecture, technologies, source files, workflow, commands, outputs, tests and limitations.
>
> **Technology stack:** Python, local TF-IDF-style lexical retrieval, standard-library text processing, optional pypdf for PDFs, and unittest.


**Status:** Implemented and verified locally and in Google Colab. Fourteen unit tests pass in both validation runs, Python compilation passes, and the three-query retrieval smoke evaluation returned Recall@3 = 1.0 and MRR@3 = 1.0. See VALIDATION_REPORT.json.

## Objective

Ingest approved, non-identifying healthcare education material and retrieve relevant, source-cited passages for a user query. This offline implementation uses a lightweight local TF-IDF/cosine index and extractive evidence selection rather than free-form medical generation.

## Features

- Markdown, plain-text and PDF ingestion. PDF ingestion uses pypdf.
- Markdown-heading-aware chunking and fixed-size overlapping chunking.
- Local sparse TF-IDF-style vectors and cosine similarity; no remote embedding service or GPU required.
- Ranked passages with source filename, section/page metadata and stable chunk citations.
- Extractive, evidence-only draft responses; abstains when no passage meets the relevance threshold.
- Small synthetic, non-identifying demonstration corpus and a retrieval evaluation harness.
- Unit tests for source loading, chunking, ranking, citations, invalid inputs and abstention.
- Google Colab verification: 14 tests passed, the retrieval evaluation completed, and a sample query returned a citation.

## Quick start

Python 3.10+ is recommended. Only PDF ingestion needs the extra pypdf package; Markdown/text retrieval uses the Python standard library.

~~~bash
python -m venv .venv
# Windows: .venv/Scripts/activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.evaluate
python -m src.cli "What information should I prepare before an appointment?"
python -m src.cli "What should I do before taking an unfamiliar medicine?" --json
~~~

## Google Colab

~~~python
!git clone https://github.com/chinthavishnupriya/ml-genai-enterprise-lab.git
%cd /content/ml-genai-enterprise-lab/05-healthcare-rag
!python -m pip install -r requirements.txt
!python -m unittest discover -s tests -v
!python -m src.evaluate
!python -m src.cli "What information should I prepare before an appointment?"
~~~

## Retrieval strategy

The default heading strategy preserves Markdown section titles. Use --strategy fixed in the CLI when comparing fixed-size-overlap chunking. Results are ranked with a local term-weighted vector representation and cosine similarity. The answer builder selects sentences only from retrieved chunks and prints citations for the source chunks.

## Safety, privacy and limitations

- This is an educational information-retrieval prototype, not a diagnostic, triage, or clinical decision tool. Generated snippets do not constitute medical advice.
- The bundled documents are synthetic demonstrations, not authoritative clinical references. Do not make medical decisions from them.
- Do not ingest patient-identifiable information, patient records, private messages, credentials or confidential documents.
- A source citation means that a retrieved chunk supports the displayed text; it does not prove that the underlying document is correct or current.
- The token-based index may miss synonyms and complex language. Its confidence is not medically calibrated. A small retrieval test set is not clinical validation.
- Use authoritative, current, permissioned materials and have qualified clinical and privacy reviewers assess any real deployment. For urgent or potentially life-threatening symptoms, contact local emergency services rather than relying on this tool.
