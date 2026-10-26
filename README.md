# Machine Learning & Generative AI — Enterprise Lab

A 10-project hands-on lab based on the supplied Real-World ML & GenAI problem-statement handbook. Each project has its own numbered directory, README, reproducible code, and tests where applicable.

## Progress

- **Project 01 — Industrial IoT Failure Prediction:** implemented and re-audited. Six automated tests pass. Regression and classification use grouped train/validation/test splits; the failure threshold is selected using validation data only and sensor imputation is learned from training rows only.
- **Project 02 — Fintech Fraud Detection & Customer Segmentation:** implemented and re-audited. Seven automated tests pass. Decision Tree and Random Forest are compared on held-out test data, and K-Means generates customer segments. Numeric imputation is learned from training rows only.
- **Project 03 — Supply Chain Demand Forecasting:** implemented and verified in Google Colab. Ten automated tests pass; full end-to-end random search and holdout evaluation completed. Includes a PyTorch MLP, hyperparameter tuning, and a Random Forest baseline. In the recorded Colab run, Random Forest outperformed the MLP.
- **Project 04 — Customer Support Intent Router:** implemented and verified locally and in Google Colab. Thirteen automated tests pass, and an 18-ticket handcrafted synthetic benchmark is included; its score is not a production accuracy estimate.
- **Project 05 — Healthcare Clinical Manual Ingestion & RAG:** implemented and verified locally and in Google Colab. Fourteen unit tests pass; source-cited extractive retrieval and a synthetic demonstration corpus are included. It is not a clinical decision tool.
- **Project 06 — Legal Contract Q&A with Hybrid Search:** implemented and verified in Google Colab. Sixteen tests pass and the six-query fictional-contract retrieval evaluation has Recall@5 = 1.0 and MRR@5 = 1.0. These are retrieval smoke-test results, not legal correctness metrics.
- **Project 07 — Autonomous Financial Analyst Agent:** implemented and verified in Google Colab. Twenty tests pass; the four-case synthetic smoke evaluation has 100% answer coverage, and the risk-note search returns the relevant section. These results verify demo behavior, not financial correctness.
- **Project 08 — Collaborative Multi-Agent DevSecOps Audit:** implemented and verified in Google Colab. Seventeen tests pass. The intentionally insecure demo fixture produced 7 findings across 5 scanned files (4 high, 1 medium, 2 low) and a heuristic risk score of 34/100 (high). The scanner is read-only; findings require human review and are not a security certification.
- **Project 09 — Insurance Claims LoRA / QLoRA:** implemented; offline Colab verification is pending. Includes 64 synthetic instruction examples with scenario-grouped splits, structured routing outputs, a rules baseline, and optional LoRA/QLoRA fine-tuning and adapter evaluation. All outputs require human review; it does not decide coverage or payouts.
- **Project 10 — Enterprise Agentic RAG Capstone:** starter folder exists; implementation and validation are pending.

## Project index

1. [Industrial IoT Failure Prediction](01-industrial-iot-failure-prediction/)
2. [Fintech Fraud Detection](02-fintech-fraud-detection/)
3. [Supply Chain Forecasting](03-supply-chain-forecasting/)
4. [Customer Support Intent Router](04-customer-support-intent-router/)
5. [Healthcare RAG](05-healthcare-rag/)
6. [Legal Hybrid RAG](06-legal-hybrid-rag/)
7. [Financial Analyst Agent](07-financial-analyst-agent/)
8. [DevSecOps Multi-Agent Audit](08-devsecops-multi-agent/)
9. [Insurance Claims LoRA/QLoRA](09-insurance-qlora/)
10. [Enterprise Agentic RAG Capstone](10-enterprise-agentic-rag-capstone/)

## Usage

Read the README inside each project directory for dependencies, setup, execution, generated outputs, and verification notes. Most Python projects recommend Python 3.10–3.13 depending on package compatibility.

## Data, reproducibility, and safety

Projects 01–03 use synthetic demonstration datasets, Project 04 uses handcrafted synthetic support-ticket examples, Project 05 uses synthetic educational health-document examples, Project 06 uses fictional contract text, Project 07 uses invented quarterly metrics and research notes, Project 08 uses intentionally insecure non-executable scanner fixtures, and Project 09 uses fictional insurance claim examples. Metrics indicate whether documented code paths execute; they are not evidence of production accuracy, clinical performance, legal correctness, financial returns, or security certification. Do not commit private data, credentials, tokens, or local environment files. Real-world use requires representative domain-specific validation, security/privacy review, monitoring, and human oversight.
