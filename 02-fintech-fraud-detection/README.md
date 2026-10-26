# Project 02 — Fintech Transaction Fraud Scoring & High-Value Customer Segmentation

**Status: Implemented baseline; 7 unit tests and end-to-end training/segmentation passed locally.**

This project follows Problem Statement 2 in the supplied handbook: Decision Tree fraud classification, Random Forest comparison, and K-Means customer behavior segmentation using normalized profile vectors.

## Implemented

- Deterministic synthetic data generator (12,000 transactions, 1,200 customers by default).
- Data cleaning and schema validation.
- Stratified train/validation/test split. Model thresholds are chosen only on validation data; final metrics are reported on held-out test data.
- Decision Tree and Random Forest with preprocessing pipelines, balanced class weighting, unknown-category-safe one-hot encoding.
- ROC-AUC, average precision, precision, recall, F1 and confusion matrices.
- Validation threshold sweep; feature-importance CSV files.
- Customer-level aggregate profiles with fraud labels excluded from clustering.
- StandardScaler + K-Means; candidate K values evaluated with inertia (Elbow) and Silhouette Score.
- Reports and figures generated automatically.
- 7 automated unit tests.

## Quick start

Python 3.10–3.12 recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.train
python -m src.segment
```

The training script writes `data/synthetic_transactions.csv`, model files to `models/`, and classification reports/plots to `reports/`. The segmentation script writes customer cluster assignments and cluster summaries into `reports/`.

## Source code

- `src/data.py`: deterministic synthetic transactions and validation.
- `src/modeling.py`: preprocessing, model factories, threshold selection and metrics.
- `src/train.py`: model training, holdout evaluation and artifact export.
- `src/segment.py`: customer aggregation, K-Means selection and profile plots.
- `tests/`: data and metric behavior tests.

## Example run (synthetic data)

The checked local run generated 12,000 rows, 1,200 customer profiles and 746 fraud-positive transactions (6.22%). On the held-out synthetic test split, Random Forest ROC-AUC was approximately 0.788 and average precision was approximately 0.231; Decision Tree ROC-AUC was approximately 0.711. K-Means selected K=2 by the highest Silhouette Score among the tested candidates (about 0.321).

These figures are reproducibility evidence for the synthetic demonstration only, not a promise of performance on real financial data.

## Limitations and responsible use

- Data and labels are synthetic; this is not a bank-grade fraud model.
- Synthetic label generation is deliberately simplified and metrics may change if parameters or dependency versions change.
- Real deployment needs approved historical labels, temporal and customer-aware validation, leakage assessment, cost-sensitive threshold selection, model monitoring, privacy/security reviews, and human oversight.
- Do not use this project to decline transactions or make actual financial decisions.
- Never commit customer data, access tokens, or API keys.
