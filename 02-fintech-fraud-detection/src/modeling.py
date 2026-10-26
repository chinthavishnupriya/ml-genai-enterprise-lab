"""Preprocessing, model factories, threshold tuning, and evaluation helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from .data import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def make_models(seed: int = 42) -> dict[str, Pipeline]:
    """Create an interpretable tree and an ensemble tree baseline."""
    return {
        "Decision Tree": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", DecisionTreeClassifier(
                max_depth=7, min_samples_leaf=15, class_weight="balanced", random_state=seed
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=180, max_depth=14, min_samples_leaf=5,
                class_weight="balanced_subsample", n_jobs=-1, random_state=seed
            )),
        ]),
    }


def choose_threshold(y_true, probabilities, thresholds=None) -> tuple[float, pd.DataFrame]:
    """Choose a threshold maximizing validation F1; do not tune on test data."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(probabilities):
        raise ValueError("Labels and probabilities must have equal non-zero length.")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1].")
    if np.unique(y_true).size < 2:
        raise ValueError("Validation labels must contain both classes.")
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for threshold in thresholds:
        prediction = (probabilities >= float(threshold)).astype(int)
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, prediction, zero_division=0)),
            "recall": float(recall_score(y_true, prediction, zero_division=0)),
            "f1": float(f1_score(y_true, prediction, zero_division=0)),
        })
    report = pd.DataFrame(rows)
    best = report.sort_values(["f1", "recall", "precision", "threshold"],
                              ascending=[False, False, False, False]).iloc[0]
    return float(best["threshold"]), report


def evaluate_classifier(y_true, probabilities, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    prediction = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel().tolist()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if np.unique(y_true).size == 2 else None,
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "confusion_matrix_tn_fp_fn_tp": cm,
        "test_positive_rate": float(np.mean(y_true)),
    }
