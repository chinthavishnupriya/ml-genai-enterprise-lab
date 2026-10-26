# Project 03 — Supply Chain Demand Forecasting with Neural Networks & Automated Tuning

**Status:** Implemented and verified in Google Colab. Ten automated tests pass; full random-search training and holdout comparison completed.

## Problem statement

Forecast daily retail demand per SKU using a feedforward neural network. Tune hidden-layer dimensions, learning rate, dropout and L2 weight decay, then compare the selected MLP with a Random Forest regression baseline.

## Implemented

- Deterministic synthetic daily demand for 50 SKUs over 200 days, with region, promotions/discounts, price, weather, shipping delays and seasonal features.
- Historical lag-1, lag-7 and trailing 7-day mean features using prior target values only.
- Chronological train/validation/test split by date; no date appears in more than one partition.
- PyTorch dense MLP with BatchNorm and Dropout.
- Seeded random hyperparameter search with early stopping on validation loss; model initialization is seeded for reproducibility.
- Final preprocessing and model refit on train + validation data; held-out later dates are used only for final comparison.
- Random Forest baseline and regression metrics: RMSE, MAE, R² and sMAPE.
- Hyperparameter table, validation-loss curve, test comparison, prediction chart, and automated tests.
- Google Colab run outputs saved under `reports/` and `models/`.

## Quick start

Python 3.10–3.13 may work depending on the installed PyTorch build; the verified Colab run used Python 3.13 and PyTorch 2.11.0+cpu. PyTorch CPU is sufficient; no GPU or API key is required.

The project pins the CPU-only PyTorch wheel in `requirements.txt` to avoid downloading CUDA/GPU packages on CPU machines. If a previous installation timed out, rerun `python -m pip install --timeout 120 --retries 5 -r requirements.txt`. Then verify with `python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"`; the version should include `+cpu`.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.train
```

The default run performs four random-search trials with a maximum of 25 epochs each; early stopping may finish earlier. The generated data, trained models, metrics and plots are saved in `data/`, `models/`, and `reports/`.

To run a smaller quick experiment:

```bash
python -c "from src.train import main; main(n_trials=2, max_epochs=5)"
```

## Main generated outputs

- `reports/metrics.json`
- `reports/hyperparameter_search.csv` and `reports/hyperparameter_search.png`
- `reports/validation_loss_curve.png`
- `reports/test_model_comparison.csv` and `reports/test_model_comparison.png`
- `reports/test_predictions.csv` and `reports/test_predictions.png`
- `models/preprocessor.joblib`, `models/random_forest_regressor.joblib`, `models/demand_mlp_state_dict.pt`, and `models/best_config.json`
- `VALIDATION_REPORT.json` records test results and runtime information.

## Verification summary — Google Colab run

- Python compilation: PASS
- Unit tests: 10 passed, 0 failed
- Full default training run: PASS (four random-search trials)
- Generated synthetic rows: 9,650 across 50 SKUs
- Selected hyperparameters: hidden layers `[64, 32]`, dropout `0.10`, learning rate `0.003`, weight decay `0.0`, selected at epoch 22
- Runtime reported by the Colab notebook: Python 3.13, PyTorch 2.11.0+cpu, NumPy 2.1.3, Pandas 2.2.3

| Test metric | MLP | Random Forest baseline |
|---|---:|---:|
| RMSE ↓ | 25.6536 | **14.0935** |
| MAE ↓ | 19.7635 | **10.6479** |
| R² ↑ | 0.6263 | **0.8872** |
| sMAPE ↓ | 26.0713% | **13.3374%** |

The Random Forest baseline performed better on all four metrics in this Colab run. The MLP metrics differ from an earlier local reference run, while the Random Forest metrics remained the same; the validation report records the Colab environment and exact run metrics. Keep dependency/runtime versions fixed when comparing runs.

## Limitations

This is synthetic data and a one-step-ahead rolling forecast where the previous observed demand is available for lag features, including during the later holdout dates. It is not a multi-step forecast and has not been validated on real sales data. Real deployment needs time-aware backtesting, promotion-calendar checks, SKU/region coverage checks, drift monitoring, inventory-cost-aware metrics and stakeholder review.
