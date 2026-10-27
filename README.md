# Machine Learning & Generative AI — Enterprise Lab

A 10-project hands-on lab based on the supplied Real-World ML & GenAI problem-statement handbook. Each project has its own numbered directory, README, reproducible code, and tests where applicable.

## Progress

- **Project 01 — Industrial IoT Failure Prediction:** implemented and re-audited. Six automated tests pass. Regression and classification use grouped train/validation/test splits; the failure threshold is selected using validation data only and sensor imputation is learned from training rows only.
- **Project 02 — Fintech Fraud Detection & Customer Segmentation:** implemented and re-audited. Seven automated tests pass. Decision Tree and Random Forest are compared on held-out test data, and K-Means generates customer segments. Numeric imputation is learned from training rows only.
- **Project 03 — Supply Chain Demand Forecasting:** implemented and verified in Google Colab. Ten automated tests pass; full end-to-end random search and holdout evaluation completed. Includes a PyTorch MLP, hyperparameter tuning, and a Random Forest baseline. In the recorded Colab run, Random Forest outperformed the MLP.
- **Project 04 — Customer Support Intent Router:** original 13 tests and curated benchmark passed in Colab; the hardened optional provider now enforces deterministic security/high-priority rules. All 15 tests passed in GitHub Actions.
- **Project 05 — Healthcare Clinical Manual Ingestion & RAG:** implemented and verified locally and in Google Colab. Fourteen unit tests pass; source-cited extractive retrieval and a synthetic demonstration corpus are included. It is not a clinical decision tool.
- **Project 06 — Legal Contract Q&A with Hybrid Search:** implemented and verified in Google Colab. Sixteen tests pass and the six-query fictional-contract retrieval evaluation has Recall@5 = 1.0 and MRR@5 = 1.0. These are retrieval smoke-test results, not legal correctness metrics.
- **Project 07 — Autonomous Financial Analyst Agent:** implemented and verified in Google Colab. Twenty tests pass; the four-case synthetic smoke evaluation has 100% answer coverage, and the risk-note search returns the relevant section. These results verify demo behavior, not financial correctness.
- **Project 08 — Collaborative Multi-Agent DevSecOps Audit:** original 17 tests and sample fixture audit passed in Colab; direct-URL dependency detection is fixed. All 19 tests passed in GitHub Actions. The scanner is read-only and is not a security certification.
- **Project 09 — Insurance Claims LoRA / QLoRA:** guarded hybrid pipeline verified in Colab. All 30 tests pass, and guard evaluation returned 100% on seven checked fields across eight synthetic cases, even when model fields were deliberately incorrect. Raw adapter field fidelity remains weak; deterministic code supplies structured fields, and model text is limited to a screened draft summary for human verification. These results do not demonstrate real claims accuracy.
- **Project 10 — Enterprise Agentic RAG Capstone:** verified in Google Colab; all 29 tests pass. Ingestion loaded 7 synthetic shipments and 25 policy chunks, and tracking lookup, customs retrieval, and prompt-injection blocking ran successfully. After a retrieval ranking fix, the previously failing cross-border escalation evaluation case now matches its expected citation. Aggregate evaluation metrics remain unreported because they were not captured in the screenshot.

## Repository audit and automated checks

See [REPOSITORY_AUDIT.md](REPOSITORY_AUDIT.md) for the project-by-project review, limitations, and audit fixes. The GitHub Actions workflow at [.github/workflows/project-tests.yml](.github/workflows/project-tests.yml) runs each project's unit tests and validates all ten project README/JSON validation-report pairs. The first complete CI run passed all ten test jobs and the repository-metadata job: [workflow run 40463898783](https://github.com/chinthavishnupriya/ml-genai-enterprise-lab/actions/runs/40463898783).

## Project index

1. [Industrial IoT Failure Prediction](01-industrial-iot-failure-prediction/) · [Detailed technical guide](01-industrial-iot-failure-prediction/PROJECT_DOCUMENTATION.md)
2. [Fintech Fraud Detection](02-fintech-fraud-detection/) · [Detailed technical guide](02-fintech-fraud-detection/PROJECT_DOCUMENTATION.md)
3. [Supply Chain Forecasting](03-supply-chain-forecasting/) · [Detailed technical guide](03-supply-chain-forecasting/PROJECT_DOCUMENTATION.md)
4. [Customer Support Intent Router](04-customer-support-intent-router/) · [Detailed technical guide](04-customer-support-intent-router/PROJECT_DOCUMENTATION.md)
5. [Healthcare RAG](05-healthcare-rag/) · [Detailed technical guide](05-healthcare-rag/PROJECT_DOCUMENTATION.md)
6. [Legal Hybrid RAG](06-legal-hybrid-rag/) · [Detailed technical guide](06-legal-hybrid-rag/PROJECT_DOCUMENTATION.md)
7. [Financial Analyst Agent](07-financial-analyst-agent/) · [Detailed technical guide](07-financial-analyst-agent/PROJECT_DOCUMENTATION.md)
8. [DevSecOps Multi-Agent Audit](08-devsecops-multi-agent/) · [Detailed technical guide](08-devsecops-multi-agent/PROJECT_DOCUMENTATION.md)
9. [Insurance Claims LoRA/QLoRA](09-insurance-qlora/) · [Detailed technical guide](09-insurance-qlora/PROJECT_DOCUMENTATION.md)
10. [Enterprise Agentic RAG Capstone](10-enterprise-agentic-rag-capstone/) · [Detailed technical guide](10-enterprise-agentic-rag-capstone/PROJECT_DOCUMENTATION.md)

## Documentation structure

Each project directory now contains two documents: `README.md` for quick start and verification status, and `PROJECT_DOCUMENTATION.md` for the detailed technology stack, architecture, source-file map, implementation workflow, commands, generated artifacts, testing evidence, limitations and potential next steps.

## Usage

Open each project's README for the quick start, then use its detailed technical guide to understand the implementation. Most Python projects recommend Python 3.10–3.13 depending on package compatibility.

## Data, reproducibility, and safety

Projects 01–03 use synthetic demonstration datasets, Project 04 uses handcrafted synthetic support-ticket examples, Project 05 uses synthetic educational health-document examples, Project 06 uses fictional contract text, Project 07 uses invented quarterly metrics and research notes, Project 08 uses intentionally insecure non-executable scanner fixtures, Project 09 uses fictional insurance claim examples, and Project 10 uses synthetic shipment records and logistics-policy articles. Metrics indicate whether documented code paths execute; they are not evidence of production accuracy, clinical performance, legal correctness, financial returns, or security certification. Do not commit private data, credentials, tokens, or local environment files. Real-world use requires representative domain-specific validation, security/privacy review, monitoring, and human oversight.
