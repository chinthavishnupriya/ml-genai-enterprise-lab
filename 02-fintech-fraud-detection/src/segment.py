"""K-Means customer segmentation and report generation."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from .data import make_synthetic_transactions, validate_transactions

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
SEED = 42


def build_customer_profiles(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate behavior by customer; labels are deliberately excluded."""
    return transactions.groupby("customer_id", as_index=False).agg(
        transaction_count=("transaction_id", "count"),
        mean_amount=("amount", "mean"),
        median_amount=("amount", "median"),
        amount_std=("amount", "std"),
        mean_location_delta_km=("location_delta_km", "mean"),
        device_change_rate=("device_changed", "mean"),
        mean_transactions_last_hour=("transactions_last_hour", "mean"),
        account_age_days=("account_age_days", "median"),
    ).fillna(0)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATA_DIR / "synthetic_transactions.csv"
    if data_path.exists():
        transactions = validate_transactions(pd.read_csv(data_path))
    else:
        transactions = validate_transactions(make_synthetic_transactions())
        transactions.to_csv(data_path, index=False)

    profiles = build_customer_profiles(transactions)
    features = profiles.drop(columns=["customer_id"])
    feature_names = features.columns.tolist()
    scaled = StandardScaler().fit_transform(features)
    candidate_ks = list(range(2, min(9, len(profiles))))
    if not candidate_ks:
        raise ValueError("Need at least three customers for K-Means selection.")

    rows, fitted = [], {}
    for k in candidate_ks:
        model = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = model.fit_predict(scaled)
        silhouette = float(silhouette_score(scaled, labels)) if len(np.unique(labels)) > 1 else float("nan")
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette_score": silhouette})
        fitted[k] = (model, labels)
    selection = pd.DataFrame(rows)
    valid = selection.dropna(subset=["silhouette_score"])
    if valid.empty:
        raise RuntimeError("No valid Silhouette Score was available.")
    selected_k = int(valid.sort_values(["silhouette_score", "k"], ascending=[False, True]).iloc[0]["k"])
    selected_model, labels = fitted[selected_k]

    assignments = profiles.copy()
    assignments["cluster"] = labels
    assignments.to_csv(REPORT_DIR / "customer_clusters.csv", index=False)
    cluster_profiles = assignments.groupby("cluster", as_index=False).agg(
        customer_count=("customer_id", "count"),
        mean_transaction_count=("transaction_count", "mean"),
        mean_amount=("mean_amount", "mean"),
        mean_median_amount=("median_amount", "mean"),
        mean_amount_std=("amount_std", "mean"),
        mean_location_delta_km=("mean_location_delta_km", "mean"),
        mean_device_change_rate=("device_change_rate", "mean"),
        mean_transactions_last_hour=("mean_transactions_last_hour", "mean"),
        mean_account_age_days=("account_age_days", "mean"),
    )
    cluster_profiles.to_csv(REPORT_DIR / "customer_cluster_profiles.csv", index=False)
    selection.to_csv(REPORT_DIR / "cluster_selection.csv", index=False)
    selected_silhouette = float(valid.loc[valid["k"] == selected_k, "silhouette_score"].iloc[0])
    (REPORT_DIR / "segmentation_metrics.json").write_text(json.dumps({
        "dataset": "synthetic customer behavior profiles",
        "warning": "Exploratory clusters only; no real customer decision use.",
        "customer_count": int(len(profiles)),
        "features": feature_names,
        "candidate_k": candidate_ks,
        "selected_k_by_silhouette": selected_k,
        "selected_silhouette_score": selected_silhouette,
        "inertia": float(selected_model.inertia_),
    }, indent=2), encoding="utf-8")

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(selection["k"], selection["inertia"], marker="o")
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Inertia (Elbow)")
    ax2 = ax1.twinx()
    ax2.plot(selection["k"], selection["silhouette_score"], marker="s", linestyle="--")
    ax2.set_ylabel("Silhouette Score")
    plt.title("K-Means cluster selection")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "cluster_selection.png", dpi=150)
    plt.close(fig)

    plot_metrics = ["mean_amount", "mean_transaction_count", "mean_location_delta_km", "mean_device_change_rate"]
    profile_plot = cluster_profiles.set_index("cluster")[plot_metrics]
    profile_plot.columns = ["Avg amount", "Avg tx count", "Location delta", "Device-change rate"]
    ax = profile_plot.plot(kind="bar", figsize=(10, 5))
    ax.set_title(f"Customer cluster profiles (K={selected_k})")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mean value (raw units; scales differ)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "cluster_profiles.png", dpi=150)
    plt.close()
    print("Customer segmentation complete.")
    print(f"Customers: {len(profiles)} | Selected K: {selected_k}")
    print(f"Silhouette Score: {selected_silhouette:.4f}")
    print(cluster_profiles.to_string(index=False))
    print(f"Reports: {REPORT_DIR}")


if __name__ == "__main__":
    main()
