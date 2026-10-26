"""Synthetic industrial telemetry generation and input cleaning."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = [
    "vibration_hz", "spindle_temp_c", "motor_voltage_v",
    "acoustic_noise_db", "operating_hours"
]
TARGET_REG = "rul_hours"
TARGET_CLS = "failure_within_24h"

def make_synthetic_telemetry(n_machines: int = 30, rows_per_machine: int = 160,
                             seed: int = 42) -> pd.DataFrame:
    """Create deterministic synthetic telemetry for a reproducible demo."""
    if n_machines < 4 or rows_per_machine < 30:
        raise ValueError("Use at least 4 machines and 30 rows per machine.")
    rng = np.random.default_rng(seed)
    rows = []
    base_time = pd.Timestamp("2025-01-01", tz="UTC")
    for machine in range(n_machines):
        offset = rng.normal(0, 0.25)
        start_hours = rng.uniform(100, 2000)
        for step in range(rows_per_machine):
            age = step / max(rows_per_machine - 1, 1)
            operating_hours = start_hours + step * 0.5
            vibration = max(0.1, 2.0 + 2.4 * age + offset + rng.normal(0, 0.45))
            temp = 55 + 20 * age + 3.5 * offset + rng.normal(0, 3.0)
            voltage = 230 - 5.0 * age + rng.normal(0, 2.0)
            noise = 58 + 12 * age + 2.0 * offset + rng.normal(0, 2.5)
            latent_risk = -5.0 + 1.15 * vibration + 0.045 * (temp - 60) + 0.12 * (noise - 60)
            prob = 1.0 / (1.0 + np.exp(-latent_risk))
            failure = int(rng.random() < prob)
            rul = max(0.5, 220 * (1 - age) - 7 * max(vibration - 3.0, 0)
                      - 0.9 * max(temp - 65, 0) + rng.normal(0, 8))
            rows.append({
                "timestamp": (base_time + pd.Timedelta(hours=step, days=machine)).isoformat(),
                "machine_id": f"M{machine+1:03d}",
                "vibration_hz": vibration,
                "spindle_temp_c": temp,
                "motor_voltage_v": voltage,
                "acoustic_noise_db": noise,
                "operating_hours": operating_hours,
                "rul_hours": rul,
                "failure_within_24h": failure,
            })
    return pd.DataFrame(rows)

def clean_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, coerce values, drop invalid targets and impute sensors."""
    required = {"timestamp", "machine_id", *FEATURES, TARGET_REG, TARGET_CLS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    for col in FEATURES + [TARGET_REG, TARGET_CLS]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp", "machine_id", TARGET_REG, TARGET_CLS])
    out = out[out[TARGET_CLS].isin([0, 1])]
    out = out[out[TARGET_REG] >= 0]
    for col in FEATURES:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        out[col] = out[col].fillna(out[col].median())
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.reset_index(drop=True)
