"""
drift_detection.py - Compares the statistical distribution of new/incoming
data against a reference distribution, per feature, using the
Kolmogorov-Smirnov (KS) test. Flags features that have likely "drifted".
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DRIFT_P_VALUE_THRESHOLD = 0.05


def detect_drift(reference_df, new_df, feature_cols):
    results = []
    for col in feature_cols:
        if col not in new_df.columns:
            continue
        ref_values = reference_df[col].dropna()
        new_values = new_df[col].dropna()

        if len(new_values) < 2 or len(ref_values) < 2:
            continue

        stat, p_value = stats.ks_2samp(ref_values, new_values)
        drifted = p_value < DRIFT_P_VALUE_THRESHOLD

        results.append({
            "feature": col,
            "ks_statistic": round(stat, 4),
            "p_value": round(p_value, 6),
            "drift_detected": drifted,
        })

    return pd.DataFrame(results).sort_values("p_value")


def main():
    train = pd.read_parquet(PROCESSED_DIR / "FD001_train_features.parquet")

    non_feature_cols = ["unit_number", "time_cycles", "RUL", "factory_id",
                         "machine_id", "machine_type"]
    feature_cols = [c for c in train.columns if c not in non_feature_cols]

    engines = train["unit_number"].unique()
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(engines)
    half = len(shuffled) // 2
    reference_engines, comparison_engines = shuffled[:half], shuffled[half:]

    reference = train[train["unit_number"].isin(reference_engines)]
    comparison = train[train["unit_number"].isin(comparison_engines)]

    print(f"Sanity check: comparing {len(reference_engines)} vs {len(comparison_engines)} train engines (same distribution -- should show LOW drift)")
    print()

    drift_report = detect_drift(reference, comparison, feature_cols)
    n_drifted = drift_report["drift_detected"].sum()
    print(f"{n_drifted} / {len(drift_report)} features show drift (expect this to be low/near-zero)")
    print(drift_report.head(5).to_string(index=False))

    print()
    print("=" * 60)
    print("Production check: train (reference) vs test (incoming data)")
    print("=" * 60)

    test = pd.read_parquet(PROCESSED_DIR / "FD001_test_features.parquet")
    prod_drift_report = detect_drift(train, test, feature_cols)
    n_prod_drifted = prod_drift_report["drift_detected"].sum()

    print()
    print(f"{n_prod_drifted} / {len(prod_drift_report)} features show drift vs test set")
    print()
    print("Top 10 most drifted features:")
    print(prod_drift_report.head(10).to_string(index=False))

    out_path = PROCESSED_DIR / "drift_report.csv"
    prod_drift_report.to_csv(out_path, index=False)
    print()
    print(f"Saved full drift report to {out_path}")

    print()
    print("Note: train vs test drift here is expected to be non-trivial because")
    print("test engines are truncated mid-life at random points (per C-MAPSS design),")
    print("while train engines run to failure -- this is a known dataset characteristic,")
    print("not necessarily a sign of real-world sensor/data drift.")


if __name__ == "__main__":
    main()
