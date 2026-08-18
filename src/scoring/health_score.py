"""
health_score.py — Convert raw RUL predictions into a business-friendly
0-100 Health Score, plus a risk level label.
"""

import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_rul_model_latest.joblib"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RUL_CAP = 125  # must match train_model.py


def rul_to_health_score(rul: float, rul_cap: float = RUL_CAP) -> float:
    """Maps RUL (0 to rul_cap) -> Health Score (0 to 100)."""
    score = 100 * (rul / rul_cap)
    return max(0, min(100, round(score, 1)))  # clamp to [0, 100]


def risk_level(health_score: float) -> str:
    if health_score >= 70:
        return "Healthy"
    elif health_score >= 40:
        return "Moderate Risk"
    elif health_score >= 15:
        return "High Risk"
    else:
        return "Critical"


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

    X = last_cycle[feature_cols]
    predicted_rul = model.predict(X)

    result = last_cycle[["unit_number", "time_cycles"]].copy()
    result["predicted_RUL"] = predicted_rul
    result["health_score"] = result["predicted_RUL"].apply(rul_to_health_score)
    result["risk_level"] = result["health_score"].apply(risk_level)

    result = result.sort_values("health_score")

    print("Engines needing attention first (lowest health score):")
    print(result.head(15).to_string(index=False))

    print("\nRisk level distribution:")
    print(result["risk_level"].value_counts())

    out_path = PROCESSED_DIR / "FD001_health_scores.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved health scores to {out_path}")


if __name__ == "__main__":
    main()