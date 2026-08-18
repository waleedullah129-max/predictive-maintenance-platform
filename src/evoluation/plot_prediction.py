"""
plot_predictions.py — Visualize predicted vs actual RUL on the test set,
plus a residual plot to see where the model over/under-predicts.
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_rul_model_latest.joblib"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUL_CAP = 125


def main():
    bundle = joblib.load(MODEL_PATH)
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    test = pd.read_parquet(PROCESSED_DIR / "FD001_test_features.parquet")
    last_cycle = (
        test.sort_values(["unit_number", "time_cycles"])
        .groupby("unit_number")
        .tail(1)
    )

    X_test = last_cycle[feature_cols]
    y_true = last_cycle["RUL"].clip(upper=RUL_CAP)
    y_pred = model.predict(X_test)

    residuals = y_pred - y_true.values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Plot 1: Predicted vs Actual ---
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3)
    lims = [0, RUL_CAP]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual RUL (clipped)")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("Predicted vs Actual RUL (Test Set, last cycle per engine)")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Plot 2: Residuals vs Actual RUL ---
    ax = axes[1]
    ax.scatter(y_true, residuals, alpha=0.6, edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual RUL (clipped)")
    ax.set_ylabel("Residual (Predicted - Actual)")
    ax.set_title("Residuals vs Actual RUL")
    ax.grid(alpha=0.3)
    ax.text(
        0.02, 0.02,
        "Above 0 = over-predicted (risky: model thinks engine\nhas more life left than it does)",
        transform=ax.transAxes, fontsize=8, color="darkred", va="bottom"
    )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "predicted_vs_actual.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    # --- Print a quick breakdown by RUL range ---
    print("\nError breakdown by actual RUL range:")
    bins = [0, 25, 50, 75, 100, RUL_CAP]
    labels = ["0-25", "25-50", "50-75", "75-100", f"100-{RUL_CAP}"]
    bucket = pd.cut(y_true, bins=bins, labels=labels, include_lowest=True)
    err_df = pd.DataFrame({"actual_RUL": y_true, "residual": residuals, "bucket": bucket})
    summary = err_df.groupby("bucket", observed=True)["residual"].agg(
        mean_error="mean", mean_abs_error=lambda x: x.abs().mean(), count="count"
    )
    print(summary)


if __name__ == "__main__":
    main()