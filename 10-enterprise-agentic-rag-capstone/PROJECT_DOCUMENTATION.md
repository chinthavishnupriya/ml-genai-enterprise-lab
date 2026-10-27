# Project 10 — Enterprise Agentic RAG Capstone for Logistics Operations

## 1. Project overview

Combine policy retrieval and a controlled shipment lookup into one deterministic logistics-support assistant. The system ingests a local synthetic policy corpus and shipment event records, routes a question to a read-only lookup or hybrid retrieval, and returns a source-cited answer or an explicit abstention.

Despite the project title, the current reference implementation uses deterministic routing and response templates rather than a trained LLM planner. This keeps tool permissions and behavior reproducible.

## 2. Technology stack

| Technology | Purpose |
|---|---|
| Python | Ingestion, retrieval, routing, guardrails, tool wrapper and CLI |
| Python standard library | JSONL/Markdown parsing, regular expressions, counters, math and local file access |
| JSONL | Synthetic shipment events, one JSON object per line |
| Markdown | Local policy corpus |
| `unittest` | Ingestion, retrieval, guardrails, tool routing and citation regression tests |

The offline implementation requires no API key, vector database, GPU or external service.

## 3. Ingestion and corpus

`src/ingestion.py` loads Markdown policies using headings as section boundaries and validates JSONL shipment records. It checks required fields, synthetic shipment-ID format and duplicates. The policy retriever and shipment lookup are separate so operational-style record lookup is not mixed into the policy text corpus.

The recorded Colab ingestion loaded seven synthetic shipments and 25 policy chunks from six policy files.

## 4. Hybrid retrieval and tool routing

`HybridRetriever` combines normalized BM25-style scores with TF-IDF cosine similarity, then adds a phrase and section-heading bonus. The query is tokenized and common stop words and selected inflections are normalized; heading ranking was refined to resolve a cross-border escalation regression.

The agent routes tracking-intent questions containing a valid synthetic shipment ID to the read-only shipment lookup tool. Other logistics questions go to policy retrieval. Answers contain local citations. If no sufficiently relevant result is found, the agent abstains rather than guessing live status or policy.

## 5. Guardrails and observability

- Input checks reject blank/oversized requests and pattern-match prompt-injection-like instructions.
- Email addresses and phone-like strings are redacted before downstream routing.
- A citation allowlist validates that answer citations originate in the retrieved local evidence.
- Each response carries a structured trace for input checks, selected tool, result and safety outcome.
- The agent never executes source code, writes to shipment data, calls carrier APIs or changes external state.

These checks are simple heuristics and are not comprehensive PII or prompt-injection detectors.

## 6. Source map

| File | Responsibility |
|---|---|
| `src/ingestion.py` | Policy chunking, shipment validation and corpus manifest |
| `src/retrieval.py` | Tokenization, BM25-style scoring and TF-IDF cosine hybrid |
| `src/tools.py` | Validated read-only shipment lookup and tracking intent |
| `src/guardrails.py` | Injection patterns, PII redaction and citation allowlist |
| `src/models.py` | Typed data structures for chunks, hits, tool results and agent response |
| `src/agent.py` | Deterministic routing and cited answer assembly |
| `src/cli.py` | Command-line interface |
| `src/evaluate.py` | Nine hand-labelled RAG-inspired smoke cases |
| `tests/` | Ingestion/retrieval, guardrail, agent and evaluation tests |
| `VALIDATION_REPORT.json` | Colab smoke/test evidence and limitations |

## 7. Run locally

```bash
python -m unittest discover -s tests -v
python -m src.ingestion
python -m src.evaluate
python -m src.cli "Track shipment SHP-1002 and tell me its current status."
python -m src.cli "How should cross-border delays be escalated?"
```

The answers refer to the local synthetic records and policies only; the example phrase “current status” does not turn the record into live carrier tracking.

## 8. Validation

Twenty-nine tests passed in the recorded Colab run. Ingestion loaded seven synthetic shipments and 25 policy chunks, and tracking lookup, customs retrieval and prompt-injection blocking smoke tests passed. A previously failing cross-border escalation case matched its expected citation after ranking normalization and heading scoring changes.

The latest screenshot did not show complete aggregate evaluation metrics, so no aggregate metric is claimed here. The evaluator is a small deterministic, hand-labelled smoke set—not the RAGAS package or a production RAG quality study.

## 9. Limitations and future work

- The knowledge corpus and shipment events are fictional; the application is not connected to carriers or live policy systems.
- Lexical matching may miss semantic paraphrases; regex guardrails can miss injection and PII variants.
- Add larger labelled queries, independent relevance judging, and citation correctness/coverage evaluation.
- If adding an LLM planner, keep tool allowlists, untrusted-input boundaries and citation validation outside the model; require human approval for real operations.
