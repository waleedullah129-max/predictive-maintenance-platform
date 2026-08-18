"""
alert_center.py — Generates prioritized alerts from the maintenance
recommendations, for Critical and Emergency-level engines.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def generate_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Flags engines needing urgent attention as alerts, with a severity level."""
    alert_conditions = df["risk_level"].isin(["Critical", "High Risk"])
    alerts = df[alert_conditions].copy()

    def severity(row):
        if row["risk_level"] == "Critical":
            return "SEVERE"
        return "WARNING"

    alerts["severity"] = alerts.apply(severity, axis=1)
    alerts["alert_message"] = alerts.apply(
        lambda r: (
            f"Engine {r['unit_number']}: {r['risk_level']} "
            f"(Health Score {r['health_score']}) — {r['recommended_action']}"
            + (f" | Likely cause: {r['likely_failure_mode']}"
               if r['likely_failure_mode'] != "General Degradation" else "")
        ),
        axis=1,
    )
    alerts["generated_at"] = datetime.now().isoformat(timespec="seconds")

    return alerts.sort_values(["severity", "health_score"])


def main():
    recs = pd.read_csv(PROCESSED_DIR / "FD001_maintenance_recommendations.csv")
    alerts = generate_alerts(recs)

    print(f"Generated {len(alerts)} alerts "
          f"({(alerts['severity']=='SEVERE').sum()} SEVERE, "
          f"{(alerts['severity']=='WARNING').sum()} WARNING)\n")

    for _, row in alerts.iterrows():
        print(f"[{row['severity']}] {row['alert_message']}")

    out_path = PROCESSED_DIR / "FD001_alerts.csv"
    alerts.to_csv(out_path, index=False)
    print(f"\nSaved alerts to {out_path}")


if __name__ == "__main__":
    main()