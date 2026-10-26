# Project 03 — Supply Chain Demand Forecasting with Neural Networks & Automated Tuning

**Status:** Implemented and locally verified. Ten automated tests pass; full random-search training and holdout comparison completed.

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

## Quick start

Python 3.10–3.12 recommended. PyTorch CPU is sufficient; no GPU or API key is required.

The project pins the CPU-only PyTorch wheel in `requirements.txt` to avoid downloading CUDA/GPU packages on CPU machines. If a previous installation timed out, rerun `python -m pip install --timeout 120 --retries 5 -r requirements.txt`. Then verify with `python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"`; the version should include `+cpu`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
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

## Verification summary

- Python compilation: PASS
- Unit tests: 10 passed, 0 failed
- Full default training run: PASS (four random-search trials)
- Generated synthetic rows: 9,650 across 50 SKUs
- Selected hyperparameters: hidden layers `[64, 32]`, dropout `0.10`, learning rate `0.003`, weight decay `0.0`, selected at epoch 25
- Test-set MLP RMSE: approximately 15.59 demand units
- Test-set Random Forest RMSE: approximately 14.09 demand units

The Random Forest baseline performed better on RMSE in this synthetic run, although the MLP also achieved positive held-out R². These results are reported as observed; the model is not forced to win against the baseline.

## Limitations

This is synthetic data and a one-step-ahead rolling forecast where the previous observed demand is available for lag features, including during the later holdout dates. It is not a multi-step forecast and has not been validated on real sales data. Real deployment needs time-aware backtesting, promotion-calendar checks, SKU/region coverage checks, drift monitoring, inventory-cost-aware metrics and stakeholder review.
