# Project 02 — Fintech Transaction Fraud Scoring & High-Value Customer Segmentation
> **Detailed project guide:** [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) covers the architecture, technologies, source files, workflow, commands, outputs, tests and limitations.
>
> **Technology stack:** Python, NumPy, Pandas, scikit-learn (Decision Tree, Random Forest, K-Means), Matplotlib, joblib and unittest.


**Status: Implemented baseline; seven automated tests and end-to-end training/segmentation passed.**

This project follows Problem Statement 2 in the supplied handbook: Decision Tree fraud classification, Random Forest comparison, and K-Means customer behavior segmentation using normalized profile vectors.

## Implemented

- Deterministic synthetic data generator (12,000 transactions, 1,200 customers by default).
- Schema validation, invalid-row checks, and train-partition-only numeric imputation inside scikit-learn pipelines.
- Stratified train/validation/test split. Thresholds are selected only on validation data; final classification metrics are reported on held-out test data.
- Decision Tree and Random Forest with numeric/categorical preprocessing and unknown-category-safe one-hot encoding.
- ROC-AUC, average precision, precision, recall, F1 and confusion matrices.
- Validation threshold sweeps, model feature-importance exports and comparison report.
- Customer-level aggregate profiles with the fraud label excluded from clustering.
- StandardScaler + K-Means; candidate K values evaluated with inertia (Elbow) and Silhouette Score.
- CSV/JSON reports and plots.
- Seven automated unit tests.

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

The training script writes generated transactions to `data/synthetic_transactions.csv`, models to `models/`, and classifier metrics to `reports/`. The segmentation script writes customer cluster assignments, cluster profiles, cluster-selection metrics and plots.

## Main generated outputs

- `reports/classification_comparison.csv`
- `reports/classification_metrics.json`
- `reports/threshold_evaluation_decision_tree.csv`
- `reports/threshold_evaluation_random_forest.csv`
- `reports/feature_importance_decision_tree.csv`
- `reports/feature_importance_random_forest.csv`
- `reports/roc_auc_comparison.png`
- `reports/customer_clusters.csv`
- `reports/customer_cluster_profiles.csv`
- `reports/cluster_selection.csv`, `reports/cluster_selection.png`
- `reports/cluster_profiles.png`
- `models/decision_tree.joblib` and `models/random_forest.joblib`

## Example result

In a reproducible local run on the synthetic dataset, Random Forest ROC-AUC was approximately 0.788 and average precision was approximately 0.231. K-Means selected K=2 by the highest Silhouette Score among tested candidates (approximately 0.321). Dependency versions can slightly change numeric results.

These are synthetic-demo metrics only; they do not estimate performance on real financial transactions.

## Limitations and responsible use

The labels and data are synthetic and do not represent a specific payment institution. The row-level stratified split is useful for a baseline but is not a substitute for temporal/customer-aware validation in a real deployment. Real-world use requires leakage and fairness review, a cost-sensitive operating threshold, drift monitoring, privacy/security review, and human oversight. Do not use this project to decline transactions or make actual financial decisions. Never commit customer data, credentials or API keys.
