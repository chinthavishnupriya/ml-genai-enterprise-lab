# Project 01 — Industrial IoT Telemetry Anomaly Detection & Failure Prediction

## 1. Project overview

This project demonstrates two predictive-maintenance tasks from a single industrial sensor dataset:

1. **Remaining Useful Life (RUL) regression:** estimate remaining operating life in hours.
2. **Failure-risk classification:** estimate whether a machine will experience a failure within the next 24-hour shift.

It uses reproducible synthetic telemetry, so the project is suitable for learning model pipelines and evaluation practices, not for controlling real equipment.

## 2. Technology stack

| Technology | Purpose |
|---|---|
| Python | Project implementation and command-line execution |
| NumPy | Synthetic signal/noise generation and numerical operations |
| Pandas | Telemetry tables, validation and CSV/JSON report preparation |
| scikit-learn | Linear/logistic regression pipelines, preprocessing, grouped splitting and metrics |
| Matplotlib | Test-result and threshold-evaluation plots |
| joblib | Persisting fitted scikit-learn models |
| unittest | Deterministic automated tests |

## 3. Input data and features

`src/data.py` generates 4,800 synthetic rows by default across 30 machines, with timestamps, a machine identifier, five sensor/operating features, and two targets.

**Input features**
- `vibration_hz`
- `spindle_temp_c`
- `motor_voltage_v`
- `acoustic_noise_db`
- `operating_hours`

**Targets**
- `rul_hours`: non-negative remaining-life estimate.
- `failure_within_24h`: binary label.

The generator uses deterministic random seeds for repeatability. `clean_telemetry()` validates required columns and targets, parses timestamps, removes invalid target rows, and preserves missing sensor readings so imputation can be learned by the training pipeline.

## 4. Model and evaluation workflow

1. Generate and save synthetic telemetry.
2. Validate schema and target values.
3. Split by machine into train, validation and test partitions using `GroupShuffleSplit`. A machine is never shared across these sets.
4. Fit the RUL regression pipeline on training machines and calculate RMSE/MAE on the untouched test machines.
5. Fit the failure classifier on training machines.
6. Select the classification threshold from validation predictions, then evaluate the chosen threshold on test machines.
7. Save models, metrics, threshold sweep and plots.

The scikit-learn pipelines fit preprocessing from training rows only. That avoids using held-out sensor statistics when fitting imputers. The threshold sweep is a validation artifact, not a test-set tuning tool.

## 5. Source map

| File | Responsibility |
|---|---|
| `src/data.py` | Synthetic telemetry generation and input cleaning |
| `src/features.py` | Regression/classification preprocessing and model pipelines |
| `src/train.py` | Grouped splits, fitting, metrics, persistence and plots |
| `src/evaluate.py` | Validation threshold selection and classification metrics |
| `tests/test_data.py` | Generator determinism, missing-sensor handling and schema checks |
| `tests/test_evaluate.py` | Threshold/evaluation behavior |
| `VALIDATION_REPORT.json` | Recorded test/training run summary |

## 6. Run locally

Python 3.10–3.12 is recommended for the documented environment.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.train
```

The training command creates `data/`, `models/` and `reports/` as needed.

## 7. Generated artifacts

Typical output includes the synthetic input CSV, `rul_regression.joblib`, `failure_classifier.joblib`, JSON metrics, the validation threshold sweep, actual-versus-predicted RUL plot and classification threshold metrics plot. Refer to the code and validation report for exact filenames.

## 8. Validation status and interpretation

The repository's recorded validation report reports six unit tests passing and a completed synthetic end-to-end run. The report records test RUL RMSE of about 24.01 hours and MAE of about 19.13 hours. The selected classifier threshold was 0.05; synthetic test recall was high, but precision was much lower. That trade-off is shown rather than hidden.

These values depend on the generated data and do not establish industrial predictive performance.

## 9. Limitations and future improvements

- Use real, quality-checked sensor histories and verified failure labels before evaluating real-world usefulness.
- Add temporal backtesting, probability calibration, drift monitoring and cost-based threshold selection.
- Evaluate performance by machine type and operating regime.
- Add alert review and safe fallback procedures before any operational deployment.
- **Never use this demonstration to make machine-safety decisions.**
