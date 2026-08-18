"""
Feature Engineering Pipeline
Takes the labeled train/test parquet files from loader.py and generates
rolling statistics, trend, and engineered features per engine (machine).
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ACTIVE_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

ROLLING_WINDOW = 5  # cycles


def add_rolling_features(df: pd.DataFrame, sensors: list, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Rolling mean / std / trend slope per engine, per sensor."""
    df = df.sort_values(["unit_number", "time_cycles"]).copy()
    grouped = df.groupby("unit_number")

    for sensor in sensors:
        df[f"{sensor}_roll_mean"] = grouped[sensor].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"{sensor}_roll_std"] = grouped[sensor].transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )
        df[f"{sensor}_trend"] = grouped[sensor].transform(
            lambda x: x.diff(window).fillna(0)
        )

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-style engineered features matching the project brief.
    NOTE: utilization_rate was removed -- it was computed using RUL
    (the prediction target), which leaked the answer into the features."""
    df = df.copy()

    vib_related = ["sensor_9", "sensor_14"]
    df["vibration_energy_proxy"] = df[vib_related].pow(2).sum(axis=1)

    df["temperature_gradient"] = df["sensor_4"] - df["sensor_2"]

    df["operating_hours"] = df["time_cycles"]

    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rolling_features(df, ACTIVE_SENSORS)
    df = add_engineered_features(df)
    return df


if __name__ == "__main__":
    processed_dir = PROJECT_ROOT / "data" / "processed"

    train = pd.read_parquet(processed_dir / "FD001_train_labeled.parquet")
    test = pd.read_parquet(processed_dir / "FD001_test_labeled.parquet")

    print("Before feature engineering:")
    print("  Train columns:", train.shape[1])

    train_feat = build_feature_table(train)
    test_feat = build_feature_table(test)

    print("\nAfter feature engineering:")
    print("  Train columns:", train_feat.shape[1])
    print("  New columns added:", train_feat.shape[1] - train.shape[1])
    print("\nSample new features:")
    sample_cols = ["unit_number", "time_cycles", "RUL",
                   "sensor_2_roll_mean", "sensor_2_roll_std", "sensor_2_trend",
                   "vibration_energy_proxy", "temperature_gradient", "operating_hours"]
    print(train_feat[sample_cols].head(8))

    train_feat.to_parquet(processed_dir / "FD001_train_features.parquet", index=False)
    test_feat.to_parquet(processed_dir / "FD001_test_features.parquet", index=False)
    print("\nSaved feature-engineered files to data/processed/")