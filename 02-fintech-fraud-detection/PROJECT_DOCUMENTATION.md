# Project 02 — Fintech Transaction Fraud Scoring & Customer Segmentation

## 1. Project overview

This project combines two workflows: transaction-level fraud-risk classification and customer-level behavioral segmentation. It compares an interpretable Decision Tree with a Random Forest, then uses K-Means to cluster customers from aggregated transaction behavior. All input records are synthetic.

## 2. Technology stack

| Technology | Purpose |
|---|---|
| Python | Application and CLI scripts |
| NumPy | Synthetic values and numerical processing |
| Pandas | Transaction data, customer profiles and reporting |
| scikit-learn | ColumnTransformer/Pipelines, imputers, scalers, one-hot encoding, Decision Tree, Random Forest and K-Means |
| Matplotlib | Model-comparison and segmentation plots |
| joblib | Saving trained classifiers |
| unittest | Unit tests for data and metrics |

## 3. Synthetic dataset

The default generator creates 12,000 transactions and 1,200 fictional customers. Features include transaction amount, location distance, device-change signal, recent transaction velocity, account age, hour of day and device identifier. The target is `is_fraud`.

`src/data.py` validates required columns, drops invalid labels, checks ranges and retains missing numeric feature values. It does not fill those values using the full dataset.

## 4. Fraud-classification pipeline

1. Generate and validate transactions.
2. Perform a stratified train/validation/test split (60/20/20).
3. Fit each scikit-learn pipeline on training data only. Numeric values use median imputation and scaling; categorical device IDs use most-frequent imputation and unknown-category-safe one-hot encoding.
4. Select a threshold using validation probabilities.
5. Calculate ROC-AUC, average precision, precision, recall, F1 and confusion matrix on held-out test data.
6. Save model, threshold-sweep, feature-importance, comparison and metric artifacts.

The models use class weighting to account for the imbalanced synthetic target. Threshold choice is important: a lower threshold may catch more fraud while creating more false positives and manual reviews.

## 5. Customer segmentation

`src/segment.py` aggregates transactions by customer into features such as transaction count, average/median amount, amount variability, location distance, device-change rate, transaction velocity and account age. The fraud label is intentionally excluded from clustering. The aggregate features are standardized, K-Means is fit for candidate K values, and the best candidate is selected by Silhouette Score (with inertia also reported).

## 6. Source map

| File | Responsibility |
|---|---|
| `src/data.py` | Synthetic transactions and input validation |
| `src/modeling.py` | Preprocessing, model factories, threshold choice and metrics |
| `src/train.py` | Split, fit, compare classifiers and save reports |
| `src/segment.py` | Customer aggregation, K-Means search and cluster reports |
| `tests/test_data.py` | Generator determinism, schema and missing-value tests |
| `tests/test_modeling.py` | Threshold and metric tests |
| `VALIDATION_REPORT.json` | Recorded validation and one local metric run |

## 7. Run locally

```bash
python -m venv .venv
# Activate the environment using the command for your operating system
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.train
python -m src.segment
```

## 8. Main outputs

Classification outputs include `reports/classification_comparison.csv`, `reports/classification_metrics.json`, threshold sweeps, feature-importance files and an ROC-AUC chart. Segmentation writes `reports/customer_clusters.csv`, `reports/customer_cluster_profiles.csv`, `reports/cluster_selection.csv`, cluster metrics and plots. Models are saved in `models/`.

## 9. Recorded validation

Seven unit tests, classifier training and customer segmentation were recorded as passing. One reported synthetic run selected K=2, with a Silhouette Score of about 0.321. The Random Forest test ROC-AUC was about 0.788 and average precision about 0.231. These figures illustrate behavior on this generator only, not real financial fraud performance.

## 10. Limitations and future improvements

- The row-level stratified split can share customers across partitions; production evaluation should add temporal and customer-aware holdouts.
- Evaluate with institution-specific fraud costs, analyst review capacity and false-positive costs.
- Add fairness, data privacy, drift and model-monitoring checks.
- Replace synthetic data with authorized, carefully governed data before assessing practical use.
- Do not use these demonstration results to decline a real transaction or make actual financial decisions.
