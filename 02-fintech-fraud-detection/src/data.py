"""Deterministic synthetic transaction generation and validation."""
from __future__ import annotations
import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "amount", "location_delta_km", "device_changed", "transactions_last_hour",
    "account_age_days", "hour_of_day",
]
CATEGORICAL_FEATURES = ["device_id"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_fraud"
REQUIRED_COLUMNS = {"transaction_id", "customer_id", *FEATURES, TARGET}


def make_synthetic_transactions(n_customers: int = 1200, n_transactions: int = 12000,
                                seed: int = 42) -> pd.DataFrame:
    """Create a reproducible toy dataset; it does not simulate a specific institution."""
    if n_customers < 20 or n_transactions < 100:
        raise ValueError("Use at least 20 customers and 100 transactions.")
    rng = np.random.default_rng(seed)
    customer_ids = np.array([f"C{idx:05d}" for idx in range(1, n_customers + 1)])
    customer_age = rng.integers(1, 3650, size=n_customers)
    customer_spend_scale = rng.lognormal(mean=3.25, sigma=0.65, size=n_customers)
    usual_device = rng.integers(0, 40, size=n_customers)
    customer_velocity = rng.uniform(0.4, 2.4, size=n_customers)
    customer_idx = rng.integers(0, n_customers, size=n_transactions)
    amount = rng.lognormal(mean=np.log(customer_spend_scale[customer_idx] + 1), sigma=0.8)
    age_days = customer_age[customer_idx]
    device_changed = (rng.random(n_transactions) < (0.025 + 0.10 * (age_days < 60))).astype(int)
    current_device = []
    for idx, changed in zip(customer_idx, device_changed):
        if changed:
            current_device.append(f"new-device-{int(rng.integers(0, 80)):02d}")
        else:
            current_device.append(f"trusted-device-{usual_device[idx]:02d}")
    location_delta = np.clip(rng.exponential(scale=3.5, size=n_transactions), 0, 150)
    velocity = np.clip(rng.poisson(lam=customer_velocity[customer_idx]), 0, 15)
    hour = rng.integers(0, 24, size=n_transactions)
    logit = (
        -5.80
        + 0.90 * np.log1p(amount / 25)
        + 2.40 * device_changed
        + 0.55 * np.log1p(location_delta)
        + 0.55 * velocity
        + 1.80 * (age_days < 30).astype(float)
        + 0.80 * np.isin(hour, [0, 1, 2, 3, 4]).astype(float)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    is_fraud = (rng.random(n_transactions) < probability).astype(int)
    return pd.DataFrame({
        "transaction_id": [f"T{idx:07d}" for idx in range(1, n_transactions + 1)],
        "customer_id": customer_ids[customer_idx],
        "amount": np.round(amount, 2),
        "location_delta_km": np.round(location_delta, 3),
        "device_id": current_device,
        "device_changed": device_changed,
        "transactions_last_hour": velocity,
        "account_age_days": age_days,
        "hour_of_day": hour,
        TARGET: is_fraud,
    })


def validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and sanitize values without fabricating missing fraud labels."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = df.copy()
    for col in NUMERIC_FEATURES + [TARGET]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["transaction_id", "customer_id", TARGET])
    out = out[out[TARGET].isin([0, 1])]
    out = out.replace([np.inf, -np.inf], np.nan)
    # Numeric NaNs remain until a model pipeline fits an imputer on training rows.
    # Pre-split median imputation would leak test-set feature statistics into training.
    out["device_id"] = out["device_id"].fillna("unknown-device").astype(str)
    out = out[out["amount"].isna() | (out["amount"] >= 0)]
    out = out[out["location_delta_km"].isna() | (out["location_delta_km"] >= 0)]
    out = out[out["transactions_last_hour"].isna() | (out["transactions_last_hour"] >= 0)]
    out = out[out["account_age_days"].isna() | (out["account_age_days"] >= 0)]
    out = out[out["hour_of_day"].isna() | out["hour_of_day"].between(0, 23)]
    out[TARGET] = out[TARGET].astype(int)
    if out.empty:
        raise ValueError("No valid transactions remain after cleaning.")
    if out[TARGET].nunique() < 2:
        raise ValueError("Training data must contain both fraud and non-fraud labels.")
    return out.reset_index(drop=True)
