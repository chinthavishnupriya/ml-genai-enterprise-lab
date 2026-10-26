"""Random-search a BatchNorm/Dropout MLP and compare it with Random Forest."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from .data import make_synthetic_demand, validate_demand_data, split_by_date, FEATURES, TARGET
from .modeling import (make_preprocessor, make_search_configs, train_trial, fit_final_model,
                       predict_demand, regression_metrics)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, MODEL_DIR, REPORT_DIR = ROOT / "data", ROOT / "models", ROOT / "reports"
SEED = 42


def main(n_trials: int = 4, max_epochs: int = 25):
    for folder in (DATA_DIR, MODEL_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    data = validate_demand_data(make_synthetic_demand())
    data.to_csv(DATA_DIR / "synthetic_demand.csv", index=False)
    train, validation, test = split_by_date(data)
    x_train, x_val, x_test = train[FEATURES], validation[FEATURES], test[FEATURES]
    y_train_log = np.log1p(train[TARGET].to_numpy(dtype=float))
    y_val_log = np.log1p(validation[TARGET].to_numpy(dtype=float))

    tuning_preprocessor = make_preprocessor()
    x_train_t = tuning_preprocessor.fit_transform(x_train)
    x_val_t = tuning_preprocessor.transform(x_val)
    trials = []
    best = None
    best_val_rmse = float("inf")
    configs = make_search_configs(SEED, n_trials)
    for trial_number, config in enumerate(configs, start=1):
        model, history, best_epoch = train_trial(
            x_train_t, y_train_log, x_val_t, y_val_log, config,
            max_epochs=max_epochs, patience=8, seed=SEED + trial_number,
        )
        val_prediction = predict_demand(model, x_val_t)
        metrics = regression_metrics(validation[TARGET], val_prediction)
        trials.append({
            "trial": trial_number,
            "hidden_layers": json.dumps(config["hidden_layers"]),
            "dropout": config["dropout"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "best_epoch": best_epoch,
            "validation_rmse": metrics["rmse"],
            "validation_mae": metrics["mae"],
            "validation_r2": metrics["r2"],
            "validation_smape_percent": metrics["smape_percent"],
        })
        if metrics["rmse"] < best_val_rmse:
            best_val_rmse = metrics["rmse"]
            best = {"config": config, "best_epoch": best_epoch, "trial": trial_number, "history": history}
        print(f"Trial {trial_number}/{n_trials}: validation RMSE={metrics['rmse']:.3f}; best epoch={best_epoch}")

    tuning_df = pd.DataFrame(trials).sort_values("validation_rmse")
    tuning_df.to_csv(REPORT_DIR / "hyperparameter_search.csv", index=False)
    if best is None:
        raise RuntimeError("Random search did not produce a model.")

    # Refit preprocessing and both final models using train + validation only.
    trainval = pd.concat([train, validation], ignore_index=True)
    x_trainval = trainval[FEATURES]
    y_trainval = trainval[TARGET].to_numpy(dtype=float)
    preprocessor = make_preprocessor()
    x_trainval_t = preprocessor.fit_transform(x_trainval)
    x_test_t = preprocessor.transform(x_test)
    mlp, _ = fit_final_model(
        x_trainval_t, np.log1p(y_trainval), best["config"], best["best_epoch"], seed=SEED
    )
    mlp_predictions = predict_demand(mlp, x_test_t)

    rf = RandomForestRegressor(
        n_estimators=120, max_depth=18, min_samples_leaf=2,
        max_features=0.9, n_jobs=-1, random_state=SEED,
    )
    rf.fit(x_trainval_t, y_trainval)
    rf_predictions = np.maximum(0.0, rf.predict(x_test_t))
    mlp_metrics = regression_metrics(test[TARGET], mlp_predictions)
    rf_metrics = regression_metrics(test[TARGET], rf_predictions)

    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    joblib.dump(rf, MODEL_DIR / "random_forest_regressor.joblib")
    torch.save(mlp.state_dict(), MODEL_DIR / "demand_mlp_state_dict.pt")
    (MODEL_DIR / "best_config.json").write_text(json.dumps({
        **best["config"], "best_trial": best["trial"], "selected_epochs": best["best_epoch"],
        "target_transform": "log1p; predictions converted back with expm1",
    }, indent=2), encoding="utf-8")
    metrics = {
        "dataset": "synthetic retail demand",
        "warning": "Demonstration results only; not validated for real supply-chain decisions.",
        "rows": int(len(data)), "skus": int(data["sku_id"].nunique()),
        "date_range": {"start": str(data["date"].min().date()), "end": str(data["date"].max().date())},
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "split_dates": {
            "train_end": str(train["date"].max().date()),
            "validation_start": str(validation["date"].min().date()),
            "validation_end": str(validation["date"].max().date()),
            "test_start": str(test["date"].min().date()),
        },
        "best_trial": int(best["trial"]), "best_hyperparameters": best["config"],
        "selected_epochs": int(best["best_epoch"]),
        "test_metrics": {"MLP": mlp_metrics, "Random Forest baseline": rf_metrics},
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    history = best["history"]
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["training_loss_log_space"], label="Training loss")
    plt.plot(history["epoch"], history["validation_loss_log_space"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE on log1p(demand)")
    plt.title(f"Best MLP learning curve (trial {best['trial']})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "validation_loss_curve.png", dpi=150)
    plt.close()

    ordered = tuning_df.sort_values("validation_rmse", ascending=True)
    plt.figure(figsize=(8, 5))
    plt.bar([f"T{n}" for n in ordered["trial"]], ordered["validation_rmse"])
    plt.xlabel("Random-search trial (best first)")
    plt.ylabel("Validation RMSE (units)")
    plt.title("MLP hyperparameter search")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "hyperparameter_search.png", dpi=150)
    plt.close()

    comparison = pd.DataFrame([
        {"model": "Tuned MLP", **mlp_metrics},
        {"model": "Random Forest baseline", **rf_metrics},
    ])
    comparison.to_csv(REPORT_DIR / "test_model_comparison.csv", index=False)
    plt.figure(figsize=(7, 4.5))
    plt.bar(comparison["model"], comparison["rmse"])
    plt.ylabel("Test RMSE (lower is better)")
    plt.title("Final time-holdout comparison")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "test_model_comparison.png", dpi=150)
    plt.close()

    pd.DataFrame({
        "actual_demand": test[TARGET].to_numpy(),
        "mlp_prediction": mlp_predictions,
        "random_forest_prediction": rf_predictions,
    }).to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    plt.figure(figsize=(6, 6))
    plt.scatter(test[TARGET], mlp_predictions, alpha=0.35, label="MLP")
    plt.scatter(test[TARGET], rf_predictions, alpha=0.35, label="Random Forest")
    upper = max(float(test[TARGET].max()), float(max(mlp_predictions.max(), rf_predictions.max())))
    plt.plot([0, upper], [0, upper], linestyle="--")
    plt.xlabel("Actual demand (units)")
    plt.ylabel("Predicted demand (units)")
    plt.title("Test predictions versus actual demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "test_predictions.png", dpi=150)
    plt.close()

    print("Supply-chain demand training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts: {MODEL_DIR} and {REPORT_DIR}")


if __name__ == "__main__":
    main()
