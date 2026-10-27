# Repository Audit — ML & GenAI Enterprise Lab

**Audit date:** 2026-10-27  
**Repository:** https://github.com/chinthavishnupriya/ml-genai-enterprise-lab

## Scope and method

Reviewed the GitHub tree, project source modules, unit tests, dependency files, READMEs, validation reports, and the Colab outputs available in the conversation. This review identified concrete code/documentation issues and adds automated test execution on future pushes and pull requests. Previously observed Colab results are distinguished from new tests that remain pending CI.

This is a repository review, not a formal security audit. I did not independently rerun all ten projects in a local checkout during this review.

## Project-by-project review

| Project | Existing implementation and observed verification | Audit notes / follow-up |
|---|---|---|
| **01 — Industrial IoT Failure Prediction** | Synthetic telemetry, RUL regression, failure classifier, machine-grouped train/validation/test split. Previously recorded 6 tests and end-to-end training passed. | Split and train-only imputation are documented. Results are synthetic and are not operational safety evidence. |
| **02 — Fintech Fraud Detection** | Decision Tree / Random Forest comparison, validation-only threshold selection, customer-level K-Means. Previously recorded 7 tests and training/segmentation passed. | Row-level stratified split is a baseline only; production validation should be temporal and customer-aware, as documented. |
| **03 — Supply Chain Forecasting** | PyTorch MLP, seeded search, chronological holdout, Random Forest baseline. Previously recorded 10 tests and full Colab training passed. | The recorded validation run lists PyTorch 2.11.0+cpu while the current requirements pin 2.14.1+cpu. Keep versions fixed when comparing metrics and rerun full training on the current pin before claiming exact metric reproducibility. In the recorded run the Random Forest baseline outperformed the MLP. |
| **04 — Customer Support Intent Router** | Offline rule router and optional OpenAI-compatible adapter. Previously recorded 13 tests and curated benchmark passed. | **Fix added:** the optional provider result is post-processed so a security incident cannot be downgraded and a model cannot reduce a deterministic high/urgent priority. Two regression tests were added; the revised 15-test suite is pending CI. |
| **05 — Healthcare RAG** | Local extractive TF-IDF-style retrieval, citations, abstention and synthetic health-document examples. Previously recorded 14 tests and a three-query retrieval smoke evaluation passed. | PDF parsing depends on pypdf and was not exercised in the recorded validation. The demo is not an authoritative clinical reference or decision tool. |
| **06 — Legal Hybrid RAG** | BM25 + TF-IDF, RRF, optional sentence-transformer / cross-encoder modes. Previously recorded 16 tests and six-query retrieval evaluation passed. | **Documentation fix added:** requirements no longer imply that PDF ingestion is available; the current contract loader supports Markdown/TXT only. Retrieval scores do not establish legal correctness. |
| **07 — Financial Analyst Agent** | Read-only SQLite tool, restricted arithmetic AST, local research search and action traces. Previously recorded 20 tests and four-case smoke evaluation passed. | SQLite access is allowlisted and read-only. Outputs use fictional financials, not financial advice or live market data. |
| **08 — DevSecOps Multi-Agent Audit** | Read-only secret, static-code, configuration and dependency-hygiene scanners. Previously recorded 17 tests and the sample fixture audit passed. | **Fix added:** direct-URL requirements were skipped before the DEP002 check, making detection unreachable for direct URL lines. The detector now handles those lines before package parsing, and two regression tests were added. The revised 19-test suite is pending CI. The scanner does not query CVE/advisory databases. |
| **09 — Insurance Claims LoRA/QLoRA** | Offline workflow passed 20 tests; QLoRA-NF4 training ran for two epochs on a Tesla T4. The guarded hybrid pipeline subsequently passed 30 tests and its synthetic guard evaluation. | The raw adapter was weak on structured fields (8-example run: 50% next-step match, 12.5% claim-ID match, 37.5% missing-document match). The guarded design assigns authoritative fields deterministically; keep human review and do not use the adapter for real claims decisions. |
| **10 — Enterprise Agentic RAG Capstone** | Synthetic policy/shipment corpus, hybrid retrieval, read-only lookup, citations, abstention and safety checks. Colab showed 29 tests passed; the cross-border escalation citation subsequently matched the expected section. | Aggregate metrics from the latest evaluation were not visible and are intentionally left unreported. Retrieval and safety checks are lexical/rule-based, not production-grade RAG evaluation. |

## Cross-repository improvements

1. Added .github/workflows/project-tests.yml to install each project's requirements.txt, run its unit tests on Python 3.12, and validate the ten README/JSON validation-report pairs.
2. Hardened the optional Project 4 LLM adapter against priority/security downgrades and added regression tests.
3. Fixed Project 8 direct-URL dependency detection and added regression tests.
4. Corrected Project 6 dependency notes so PDF ingestion is not presented as an available feature.
5. Kept Project 4 and Project 8 revised test results pending until CI reports success.

## Boundaries and remaining risks

- Project datasets and domain documents are synthetic or fictional; metrics do not establish production performance.
- The Project 4 and Project 8 changes are covered by new tests, but the revised suites remain pending until the workflow completes.
- Project 3's recorded training metrics and current pinned PyTorch dependency are from different versions; repeat training under the pinned environment before making exact-reproducibility claims.
- Project 9's raw adapter must not populate authoritative claim fields; deterministic rules and human review remain mandatory.
- A LICENSE file has not been added because choosing a license changes reuse permissions and should be an explicit maintainer decision.
