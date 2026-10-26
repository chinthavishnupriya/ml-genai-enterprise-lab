"""Metrics and recall-focused threshold selection."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score)

def threshold_report(y_true, probabilities, thresholds=None, min_precision=0.20):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(probabilities) or len(y_true) == 0:
        raise ValueError("y_true and probabilities must have the same non-zero length.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for threshold in thresholds:
        pred = (probabilities >= float(threshold)).astype(int)
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        })
    report = pd.DataFrame(rows)
    eligible = report[report["precision"] >= min_precision]
    if not eligible.empty:
        selected = eligible.sort_values(["recall", "f1", "threshold"],
                                        ascending=[False, False, False]).iloc[0]
    else:
        selected = report.sort_values(["f1", "recall", "threshold"],
                                      ascending=[False, False, False]).iloc[0]
    return report, float(selected["threshold"])

def classification_metrics(y_true, probabilities, threshold):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    pred = (probabilities >= threshold).astype(int)
    result = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_true, pred, labels=[0, 1]).ravel().tolist(),
    }
    result["roc_auc"] = float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None
    return result
