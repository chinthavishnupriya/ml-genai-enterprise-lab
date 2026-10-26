# Machine Learning & Generative AI — Enterprise Lab

A 10-project hands-on lab based on the supplied Real-World ML & GenAI problem-statement handbook. Each project has its own numbered directory, README, reproducible code, and tests where applicable.

## Progress

- **Project 01 — Industrial IoT Failure Prediction:** implemented and re-audited. Six automated tests pass. Regression and classification were rerun with grouped train/validation/test splits; missing numeric values are imputed inside the training pipelines, and the failure threshold is selected using validation data only.
- **Project 02 — Fintech Fraud Detection & Customer Segmentation:** implemented and re-audited. Seven automated tests pass. Decision Tree and Random Forest are compared on a held-out test split, and K-Means selects customer segments. Numeric imputation is fit inside model pipelines using training rows only.
- **Projects 03–10:** starter folders exist; implementations and validation are pending.

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

Read the README inside each project directory for dependencies, setup, execution, generated outputs, and verification notes. Most Python projects recommend Python 3.10–3.12 and provide their own `requirements.txt`.

## Data, reproducibility, and safety

Projects 01 and 02 use synthetic demonstration data. Their reported metrics show that code paths execute and are not evidence of production-quality predictions. Do not commit private data, credentials, tokens, or local `.env` files. Any real-world use requires domain-specific validation, security/privacy review, monitoring, and human oversight.
