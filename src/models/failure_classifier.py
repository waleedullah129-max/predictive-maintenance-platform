"""
failure_classifier.py — Rule-based failure mode classification.

NOTE: FD001 has only ONE fault mode in its labels, so a real trained
classifier isn't possible from this data. This module instead applies
transparent, sensor-threshold rules informed by SHAP findings (sensor_4
and sensor_11 were consistently the top drivers of low RUL predictions),
so every classification is explainable by construction.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_rul_model_latest.joblib"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def classify_failure_mode(row: pd.Series) -> str:
    """
    Applies simple threshold rules on engineered features to suggest a
    likely failure category. Thresholds are based on percentile cutoffs
    within this dataset, not physical calibration -- meant as an
    interpretable starting point, not a certified diagnosis.
    """
    reasons = []

    if row["temperature_gradient"] > row["temperature_gradient_p90"]:
        reasons.append("Overheating Risk")

    if row["vibration_energy_proxy"] > row["vibration_energy_proxy_p90"]:
        reasons.append("Bearing / Mechanical Risk")

    if row["sensor_4_roll_std"] > row["sensor_4_roll_std_p90"]:
        reasons.append("Core Performance Instability")

    if row["sensor_11_roll_std"] > row["sensor_11_roll_std_p90"]:
        reasons.append("Pressure System Irregularity")

    if not reasons:
        return "General Degradation"

    return " + ".join(reasons)


def main():
    bundle = joblib.load(MODEL_PATH)
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    test = pd.read_parquet(PROCESSED_DIR / "FD001_test_features.parquet")
    last_cycle = (
        test.sort_values(["unit_number", "time_cycles"])
        .groupby("unit_number")
        .tail(1)
        .reset_index(drop=True)
    )

    for col in ["temperature_gradient", "vibration_energy_proxy",
                "sensor_4_roll_std", "sensor_11_roll_std"]:
        last_cycle[f"{col}_p90"] = last_cycle[col].quantile(0.90)

    last_cycle["predicted_RUL"] = model.predict(last_cycle[feature_cols])
    last_cycle["likely_failure_mode"] = last_cycle.apply(classify_failure_mode, axis=1)

    result = last_cycle[["unit_number", "time_cycles", "predicted_RUL", "likely_failure_mode"]]
    result = result.sort_values("predicted_RUL")

    print("Failure mode classification (lowest RUL first):")
    print(result.head(15).to_string(index=False))

    print("\nFailure mode distribution:")
    print(result["likely_failure_mode"].value_counts())

    out_path = PROCESSED_DIR / "FD001_failure_modes.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved failure classifications to {out_path}")


if __name__ == "__main__":
    main()