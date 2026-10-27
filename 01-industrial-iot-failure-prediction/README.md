# Project 01 — Industrial IoT Telemetry Anomaly Detection & Failure Prediction
> **Detailed project guide:** [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) covers the architecture, technologies, source files, workflow, commands, outputs, tests and limitations.
>
> **Technology stack:** Python, NumPy, Pandas, scikit-learn pipelines, Matplotlib, joblib and unittest.


## Problem statement

Build a dual-model baseline that estimates Remaining Useful Life (RUL) in hours with Linear Regression and predicts a binary flag for failure within the next 24-hour operational shift with Logistic Regression.

## Implemented

- Deterministic synthetic sensor telemetry generation (4,800 rows across 30 machines by default).
- Sensor-data cleaning, polynomial feature engineering and scaling.
- Three-way machine-grouped train/validation/test split; a machine appears in only one partition.
- RUL regression evaluated on the held-out test machines with RMSE and MAE.
- Failure-classifier threshold selected from validation predictions and final metrics reported on untouched test machines.
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

The training script writes generated data to `data/`, trained models to `models/`, and metrics/plots to `reports/`. The decision-threshold sweep is saved as `reports/threshold_evaluation_validation.csv`. Missing sensor readings are imputed inside the model pipelines using training-partition statistics only.

## Safety and limitations

The example uses synthetic data only. Metrics are demonstration results, not validated production performance. Do not use this model to make real machine-safety decisions. Real deployment requires quality-checked historical labels, temporal validation, calibration, operational monitoring, human review, and safety engineering.
