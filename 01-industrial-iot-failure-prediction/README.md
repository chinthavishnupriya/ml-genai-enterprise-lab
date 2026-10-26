# Project 01 — Industrial IoT Telemetry Anomaly Detection & Failure Prediction

## Problem statement

Build a dual-model baseline that estimates Remaining Useful Life (RUL) in hours with Linear Regression and predicts a binary flag for failure within the next 24-hour operational shift with Logistic Regression.

## Implemented

- Deterministic synthetic sensor telemetry generation (4,800 rows across 30 machines by default).
- Sensor-data cleaning, polynomial feature engineering and scaling.
- Machine-grouped holdout split to keep a machine's records in one partition.
- RUL evaluation with RMSE and MAE.
- Failure classification with ROC-AUC and a recall-focused decision-threshold sweep.
- Saved model artifacts, JSON/CSV metrics and plots.
- Six unit tests.

## Run locally (Python 3.10–3.12 recommended)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.train
```

The training script writes generated data to `data/`, trained models to `models/`, and metrics/plots to `reports/`.

## Safety and limitations

The example uses synthetic data only. Metrics are demonstration results, not validated production performance. Do not use this model to make real machine-safety decisions. Real deployment requires quality-checked historical labels, temporal validation, calibration, operational monitoring, human review, and safety engineering.
