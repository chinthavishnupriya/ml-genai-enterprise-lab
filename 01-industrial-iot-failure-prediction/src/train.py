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
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["machine_id"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    X_train, X_test = train[FEATURES], test[FEATURES]

    reg = regression_pipeline()
    reg.fit(X_train, train[TARGET_REG])
    reg_pred = reg.predict(X_test)
    regression_metrics = {
        "rmse_hours": float(np.sqrt(mean_squared_error(test[TARGET_REG], reg_pred))),
        "mae_hours": float(mean_absolute_error(test[TARGET_REG], reg_pred)),
        "test_rows": int(len(test)),
        "test_machines": int(test["machine_id"].nunique()),
    }

    clf = classification_pipeline()
    clf.fit(X_train, train[TARGET_CLS])
    probabilities = clf.predict_proba(X_test)[:, 1]
    sweep, threshold = threshold_report(test[TARGET_CLS], probabilities)
    cls_metrics = classification_metrics(test[TARGET_CLS], probabilities, threshold)

    joblib.dump(reg, MODELS / "rul_regression.joblib")
    joblib.dump(clf, MODELS / "failure_classifier.joblib")
    sweep.to_csv(REPORTS / "threshold_evaluation.csv", index=False)
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
    plt.title("Failure classifier threshold evaluation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS / "threshold_metrics.png", dpi=140)
    plt.close()
    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts: {MODELS} and {REPORTS}")

if __name__ == "__main__":
    main()
