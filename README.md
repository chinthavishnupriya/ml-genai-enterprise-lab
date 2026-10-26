# Machine Learning & Generative AI — Enterprise Lab

A 10-project hands-on lab based on the supplied Real-World ML & GenAI problem-statement handbook. Each project has its own numbered directory, README, reproducible code, and tests where applicable.

## Progress

- **Project 01 — Industrial IoT Failure Prediction:** implemented and re-audited. Six automated tests pass. Regression and classification use grouped train/validation/test splits; the failure threshold is selected using validation data only and sensor imputation is learned from training rows only.
- **Project 02 — Fintech Fraud Detection & Customer Segmentation:** implemented and re-audited. Seven automated tests pass. Decision Tree and Random Forest are compared on held-out test data, and K-Means generates customer segments. Numeric imputation is learned from training rows only.
- **Project 03 — Supply Chain Demand Forecasting:** implemented and verified in Google Colab. Ten automated tests pass; full end-to-end random search and holdout evaluation completed. Includes a PyTorch MLP, hyperparameter tuning, and a Random Forest baseline. In the recorded Colab run, Random Forest outperformed the MLP.
- **Project 04 — Customer Support Intent Router:** implemented and verified. Thirteen automated tests pass, the offline CLI smoke test passes, and the curated 18-ticket benchmark has 100% category/priority agreement. This small synthetic benchmark is not a production accuracy estimate. Optional OpenAI-compatible routing is tested with a mock transport; no live API call is needed for the default route.
- **Projects 05–10:** starter folders exist; implementations and validation are pending.

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

Projects 01–03 use synthetic demonstration datasets, and Project 04 uses handcrafted synthetic support-ticket examples for its rule benchmark. Their metrics show that the documented code paths execute; they are not evidence of production-quality predictions. Do not commit private data, credentials, tokens, or local .env files. Real-world use requires representative domain-specific validation, security/privacy review, monitoring, and human oversight.
