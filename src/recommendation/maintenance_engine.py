"""
maintenance_engine.py — Combines Health Score + Failure Mode into a
final, actionable maintenance recommendation per engine.
"""

import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_rul_model_latest.joblib"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RUL_CAP = 125


def rul_to_health_score(rul: float, rul_cap: float = RUL_CAP) -> float:
    score = 100 * (rul / rul_cap)
    return max(0, min(100, round(score, 1)))


def recommend_action(health_score: float, failure_mode: str) -> str:
    """Maps health score (+failure context) to a concrete maintenance action."""
    if health_score < 15:
        return "Emergency Shutdown"
    elif health_score < 40:
        if failure_mode != "General Degradation":
            return f"Immediate Repair ({failure_mode})"
        return "Immediate Repair"
    elif health_score < 70:
        return "Scheduled Maintenance"
    else:
        return "Continue Monitoring"


def main():
    # Load health scores and failure modes (already computed by earlier scripts)
    health_df = pd.read_csv(PROCESSED_DIR / "FD001_health_scores.csv")
    failure_df = pd.read_csv(PROCESSED_DIR / "FD001_failure_modes.csv")

    merged = health_df.merge(
        failure_df[["unit_number", "likely_failure_mode"]],
        on="unit_number", how="left"
    )

    merged["recommended_action"] = merged.apply(
        lambda row: recommend_action(row["health_score"], row["likely_failure_mode"]),
        axis=1
    )

    result = merged[[
        "unit_number", "time_cycles", "predicted_RUL",
        "health_score", "risk_level", "likely_failure_mode", "recommended_action"
    ]].sort_values("health_score")

    print("Maintenance recommendations (most urgent first):")
    print(result.head(15).to_string(index=False))

    print("\nRecommended action distribution:")
    print(result["recommended_action"].value_counts())

    out_path = PROCESSED_DIR / "FD001_maintenance_recommendations.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved maintenance recommendations to {out_path}")


if __name__ == "__main__":
    main()