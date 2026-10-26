"""Train and compare Decision Tree and Random Forest fraud classifiers."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from .data import make_synthetic_transactions, validate_transactions, FEATURES, TARGET
from .modeling import make_models, choose_threshold, evaluate_classifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SEED = 42


def main() -> None:
    for directory in (DATA_DIR, MODEL_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    data = validate_transactions(make_synthetic_transactions())
    data.to_csv(DATA_DIR / "synthetic_transactions.csv", index=False)
    X, y = data[FEATURES], data[TARGET]

    # Stratified 60/20/20 split. Threshold tuning uses validation only; test is held out.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=SEED, stratify=y_train_val
    )

    comparison = []
    all_metrics = {
        "dataset": "synthetic demonstration transactions",
        "warning": "Not validated for actual financial fraud detection.",
        "row_count": int(len(data)),
        "customer_count": int(data["customer_id"].nunique()),
        "fraud_count": int(data[TARGET].sum()),
        "fraud_rate": float(data[TARGET].mean()),
        "split_rows": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "models": {},
    }
    model_scores = {}
    for name, model in make_models(SEED).items():
        model.fit(X_train, y_train)
        validation_prob = model.predict_proba(X_val)[:, 1]
        threshold, threshold_report = choose_threshold(y_val, validation_prob)
        test_prob = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_classifier(y_test, test_prob, threshold)
        all_metrics["models"][name] = metrics
        model_scores[name] = metrics["roc_auc"]
        comparison.append({"model": name, **metrics})
        slug = name.lower().replace(" ", "_")
        threshold_report.to_csv(REPORT_DIR / f"threshold_evaluation_{slug}.csv", index=False)
        joblib.dump(model, MODEL_DIR / f"{slug}.joblib")
        transformed_names = model.named_steps["preprocess"].get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_
        pd.DataFrame({"feature": transformed_names, "importance": importances}) \
            .sort_values("importance", ascending=False) \
            .to_csv(REPORT_DIR / f"feature_importance_{slug}.csv", index=False)

    comparison_df = pd.DataFrame(comparison).sort_values("roc_auc", ascending=False)
    comparison_df.to_csv(REPORT_DIR / "classification_comparison.csv", index=False)
    (REPORT_DIR / "classification_metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )
    names = list(model_scores)
    values = [model_scores[name] for name in names]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(names, values)
    plt.ylim(0, 1)
    plt.ylabel("ROC-AUC (test split)")
    plt.title("Fraud classifier comparison")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "roc_auc_comparison.png", dpi=150)
    plt.close()
    print("Fraud classification training complete.")
    print(json.dumps(all_metrics, indent=2))
    print(f"Comparison: {REPORT_DIR / 'classification_comparison.csv'}")
    print(f"Artifacts: {MODEL_DIR} and {REPORT_DIR}")


if __name__ == "__main__":
    main()
