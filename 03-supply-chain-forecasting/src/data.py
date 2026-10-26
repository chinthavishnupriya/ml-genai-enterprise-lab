"""Synthetic retail demand data with strictly historical lag features."""
from __future__ import annotations
import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "promo_discount_pct", "promo_active", "unit_price", "regional_weather_index",
    "shipping_delay_days", "day_of_week", "month", "lag_1_demand",
    "lag_7_demand", "rolling_7d_demand",
]
CATEGORICAL_FEATURES = ["sku_id", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "demand_units"
REQUIRED_COLUMNS = {"date", "sku_id", "region", *FEATURES, TARGET}
REGIONS = ["North", "South", "East", "West", "Central"]
REGION_FACTOR = {"North": 1.00, "South": 1.13, "East": 0.95, "West": 0.91, "Central": 1.06}


def make_synthetic_demand(n_skus: int = 50, n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """Build reproducible SKU/day demand data; no external downloads are required."""
    if n_skus < 4 or n_days < 30:
        raise ValueError("Use at least 4 SKUs and 30 days to support a meaningful split.")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    frames = []
    for sku_idx in range(n_skus):
        sku_id = f"SKU{sku_idx + 1:04d}"
        region = REGIONS[sku_idx % len(REGIONS)]
        base_demand = float(rng.uniform(22, 115))
        base_price = float(rng.uniform(8, 160))
        sku_phase = float(rng.uniform(-0.8, 0.8))
        rows = []
        for day_idx, date in enumerate(dates):
            promo_active = int(rng.random() < 0.13)
            discount = float(rng.uniform(0.05, 0.30)) if promo_active else 0.0
            unit_price = base_price * (1.0 - discount)
            weather = float(0.5 * np.sin(2 * np.pi * day_idx / 43 + REGIONS.index(region))
                            + rng.normal(0, 0.55))
            delay = int(np.clip(rng.poisson(0.8 + 0.3 * max(weather, 0)), 0, 7))
            weekly_factor = 1.0 + 0.15 * np.sin(2 * np.pi * day_idx / 7 + sku_phase)
            month_factor = 1.0 + 0.16 * np.sin(2 * np.pi * day_idx / 30 + REGIONS.index(region) * 0.25)
            trend_factor = 1.0 + 0.0015 * day_idx
            promo_factor = 1.0 + 1.9 * discount * promo_active
            price_factor = np.exp(-0.20 * ((unit_price / base_price) - 1.0))
            delay_factor = max(0.68, 1.0 - 0.045 * delay)
            expected = (base_demand * REGION_FACTOR[region] * weekly_factor * month_factor
                        * trend_factor * promo_factor * price_factor * delay_factor
                        * np.exp(0.055 * weather))
            demand = int(rng.poisson(max(expected, 0.1)))
            rows.append({
                "date": date,
                "sku_id": sku_id,
                "region": region,
                "promo_discount_pct": discount,
                "promo_active": promo_active,
                "unit_price": round(unit_price, 3),
                "regional_weather_index": round(weather, 4),
                "shipping_delay_days": delay,
                "day_of_week": int(date.dayofweek),
                "month": int(date.month),
                TARGET: demand,
            })
        frame = pd.DataFrame(rows).sort_values("date")
        past = frame[TARGET].shift(1)
        frame["lag_1_demand"] = past
        frame["lag_7_demand"] = frame[TARGET].shift(7)
        frame["rolling_7d_demand"] = past.rolling(7, min_periods=7).mean()
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["lag_1_demand", "lag_7_demand", "rolling_7d_demand"])
    return data.sort_values(["date", "sku_id"]).reset_index(drop=True)


def validate_demand_data(df: pd.DataFrame) -> pd.DataFrame:
    """Check schema, coerce values and preserve feature NaNs for train-only imputation."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in NUMERIC_FEATURES + [TARGET]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["date", "sku_id", "region", TARGET])
    out = out[out[TARGET] >= 0]
    out = out[out["promo_active"].isna() | out["promo_active"].isin([0, 1])]
    out = out[out["day_of_week"].isna() | out["day_of_week"].between(0, 6)]
    out = out[out["month"].isna() | out["month"].between(1, 12)]
    out[TARGET] = out[TARGET].astype(float)
    if out.empty:
        raise ValueError("No valid rows remain after cleaning.")
    return out.sort_values(["date", "sku_id"]).reset_index(drop=True)


def split_by_date(df: pd.DataFrame, train_fraction: float = 0.70,
                  validation_fraction: float = 0.15):
    """Chronological split by unique date; no date appears in multiple partitions."""
    if not (0.5 <= train_fraction < 0.9 and 0.05 <= validation_fraction < 0.4
            and train_fraction + validation_fraction < 1.0):
        raise ValueError("Invalid split fractions.")
    dates = np.array(sorted(pd.to_datetime(df["date"]).dt.normalize().unique()))
    if len(dates) < 10:
        raise ValueError("At least 10 distinct dates are needed for a chronological split.")
    train_end_idx = int(len(dates) * train_fraction) - 1
    val_end_idx = int(len(dates) * (train_fraction + validation_fraction)) - 1
    train_end = pd.Timestamp(dates[max(0, train_end_idx)])
    val_end = pd.Timestamp(dates[max(train_end_idx + 1, val_end_idx)])
    normalized_dates = pd.to_datetime(df["date"]).dt.normalize()
    train = df.loc[normalized_dates <= train_end].copy()
    validation = df.loc[(normalized_dates > train_end) & (normalized_dates <= val_end)].copy()
    test = df.loc[normalized_dates > val_end].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition.")
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)
