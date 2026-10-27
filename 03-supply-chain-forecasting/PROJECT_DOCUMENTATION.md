# Project 03 — Supply Chain Demand Forecasting with Neural Networks & Hyperparameter Search

## 1. Project overview

Forecast daily demand for multiple retail SKUs. The main model is a feedforward PyTorch multi-layer perceptron (MLP); a Random Forest regressor provides a classical baseline. A seeded random search compares hidden-layer layouts, dropout, learning rate and weight decay.

## 2. Technology stack

| Technology | Purpose |
|---|---|
| Python | Training workflow and command-line entry point |
| NumPy / Pandas | Synthetic data, lag features, metrics and tabular artifacts |
| PyTorch | MLP, batches, optimizer, loss calculation, seeded training and state-dict persistence |
| scikit-learn | Preprocessing, Random Forest baseline and regression metrics |
| Matplotlib | Search, loss and prediction plots |
| joblib | Saving preprocessing and Random Forest objects |
| unittest | Data/split/model tests |

The `requirements.txt` points pip to the PyTorch CPU wheel index and pins a CPU-only PyTorch wheel. GPU is not required for this project's default run.

## 3. Data and feature engineering

The generator creates 50 SKUs across 200 days. Synthetic factors include SKU and region, promotion state/discount, unit price, regional weather, shipping delay, weekday and month. Historical features are:
- demand at lag 1,
- demand at lag 7,
- rolling mean demand for the previous seven observations.

The lag and rolling features are computed from prior target values per SKU. The dataset generator removes rows where the initial historical window is not yet available, yielding 9,650 rows in the recorded run.

## 4. Time-aware training workflow

1. Generate and validate the demand dataset.
2. Split by unique date into chronological 70% train, 15% validation and 15% test periods. No date appears in multiple partitions.
3. Fit the preprocessing pipeline on train data; use the validation period to compare sampled MLP configurations.
4. Each MLP has dense layers, BatchNorm, ReLU and Dropout. Training uses Adam, MSE loss in log1p-demand space, fixed seeds and early stopping on validation loss.
5. Choose the best configuration by validation RMSE.
6. Refit the preprocessor on train + validation, then fit the selected MLP and Random Forest without using test targets.
7. Evaluate test predictions using RMSE, MAE, R² and sMAPE, and save models, metrics and diagnostic plots.

This is a one-step-ahead rolling forecast: for later dates the previous observed demand is available to construct lag inputs. It is not a multi-step forecast.

## 5. Source map

| File | Responsibility |
|---|---|
| `src/data.py` | Data generation, feature engineering, cleaning and date splitting |
| `src/modeling.py` | Preprocessor, MLP, search space, train/early-stop functions and metrics |
| `src/train.py` | Random search, final model fit, test comparison and artifact creation |
| `tests/test_data.py` | Data determinism, lags and chronological split |
| `tests/test_modeling.py` | Model shapes, reproducible training and metrics |
| `VALIDATION_REPORT.json` | Recorded Colab runtime, tuning and test results |

## 6. Run locally

Python 3.10–3.13 may work depending on the selected PyTorch build. Follow the CPU-only install notes in the project README.

```bash
python -m venv .venv
# Activate .venv first
python -m pip install --upgrade pip
python -m pip install --timeout 120 --retries 5 -r requirements.txt
python -m unittest discover -s tests -v
python -m src.train
```

A smaller smoke training run is available with:

```bash
python -c "from src.train import main; main(n_trials=2, max_epochs=5)"
```

## 7. Artifacts

The workflow saves synthetic demand, `models/preprocessor.joblib`, `models/random_forest_regressor.joblib`, `models/demand_mlp_state_dict.pt`, `models/best_config.json`, random-search tables and plots, validation-loss curves, test predictions and test model-comparison charts in `data/`, `models/` and `reports/`.

## 8. Recorded evaluation

The recorded Colab run passed ten tests and completed four random-search trials. In that run the Random Forest baseline outperformed the MLP on all reported metrics: Random Forest RMSE about 14.09 versus MLP RMSE about 25.65; Random Forest R² about 0.887 versus MLP R² about 0.626. This is an important comparison result: the neural network is not automatically the best model.

The recorded run used PyTorch 2.11.0+cpu, whereas current requirements pin 2.14.1+cpu. Repeat the run under the current pinned version before comparing metrics as exactly reproducible.

## 9. Limitations and future improvements

- Use real sales and inventory data with clear promotion calendars and missing-data policies.
- Add rolling-origin backtesting and multi-step forecast experiments.
- Evaluate stockout costs, service levels and inventory cost—not only point-error metrics.
- Test generalization to new SKUs and regions.
- The synthetic experiment is a learning benchmark, not a production purchasing or replenishment recommendation.
