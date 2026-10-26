"""Train and evaluate the P1 industrial IoT baseline."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .data import make_synthetic_telemetry, clean_telemetry, FEATURES, TARGET_REG, TARGET_CLS
from .features import regression_pipeline, classification_pipeline
from .evaluate import threshold_report, classification_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

def main():
    for path in (DATA, MODELS, REPORTS):
        path.mkdir(parents=True, exist_ok=True)
    raw = make_synthetic_telemetry()
    raw.to_csv(DATA / "synthetic_telemetry.csv", index=False)
    df = clean_telemetry(raw)

    # Three-way grouped split: no machine appears in more than one partition.
    # Threshold selection uses validation data; final metrics use test data only.
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_val_idx, test_idx = next(outer.split(df, groups=df["machine_id"]))
    train_val, test = df.iloc[train_val_idx], df.iloc[test_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43)
    train_idx, val_idx = next(inner.split(train_val, groups=train_val["machine_id"]))
    train, val = train_val.iloc[train_idx], train_val.iloc[val_idx]

    machine_sets = [set(part["machine_id"]) for part in (train, val, test)]
    assert not (machine_sets[0] & machine_sets[1] or machine_sets[0] & machine_sets[2]
                or machine_sets[1] & machine_sets[2]), "Machine overlap across data splits."

    X_train, X_val, X_test = train[FEATURES], val[FEATURES], test[FEATURES]

    reg = regression_pipeline()
    reg.fit(X_train, train[TARGET_REG])
    reg_pred = reg.predict(X_test)
    regression_metrics = {
        "rmse_hours": float(np.sqrt(mean_squared_error(test[TARGET_REG], reg_pred))),
        "mae_hours": float(mean_absolute_error(test[TARGET_REG], reg_pred)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_machines": int(train["machine_id"].nunique()),
        "validation_machines": int(val["machine_id"].nunique()),
        "test_machines": int(test["machine_id"].nunique()),
    }

    clf = classification_pipeline()
    clf.fit(X_train, train[TARGET_CLS])
    val_probabilities = clf.predict_proba(X_val)[:, 1]
    sweep, threshold = threshold_report(val[TARGET_CLS], val_probabilities)
    test_probabilities = clf.predict_proba(X_test)[:, 1]
    cls_metrics = classification_metrics(test[TARGET_CLS], test_probabilities, threshold)
    cls_metrics["threshold_selection"] = "validation set; test set used only for final evaluation"
    cls_metrics["train_machines"] = int(train["machine_id"].nunique())
    cls_metrics["validation_machines"] = int(val["machine_id"].nunique())
    cls_metrics["test_machines"] = int(test["machine_id"].nunique())

    joblib.dump(reg, MODELS / "rul_regression.joblib")
    joblib.dump(clf, MODELS / "failure_classifier.joblib")
    sweep.to_csv(REPORTS / "threshold_evaluation_validation.csv", index=False)
    metrics = {
        "dataset": "synthetic demonstration telemetry",
        "warning": "Not validated for real industrial decisions.",
        "rows": int(len(df)),
        "machines": int(df["machine_id"].nunique()),
        "regression": regression_metrics,
        "classification": cls_metrics,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plt.figure(figsize=(7, 5))
    plt.scatter(test[TARGET_REG], reg_pred, alpha=0.35)
    lo = min(float(test[TARGET_REG].min()), float(reg_pred.min()))
    hi = max(float(test[TARGET_REG].max()), float(reg_pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual RUL (hours)")
    plt.ylabel("Predicted RUL (hours)")
    plt.title("RUL regression: held-out machines")
    plt.tight_layout()
    plt.savefig(REPORTS / "rul_actual_vs_predicted.png", dpi=140)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(sweep["threshold"], sweep["precision"], label="Precision")
    plt.plot(sweep["threshold"], sweep["recall"], label="Recall")
    plt.plot(sweep["threshold"], sweep["f1"], label="F1")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Failure classifier threshold sweep (validation data)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS / "threshold_metrics.png", dpi=140)
    plt.close()

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts: {MODELS} and {REPORTS}")

if __name__ == "__main__":
    main()
