"""
explain_model.py — SHAP-based explainability for the RUL model.
Shows WHY the model predicted a given RUL for each engine, not just WHAT
it predicted. Satisfies the "Explainable AI" requirement.
"""

import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_rul_model_latest.joblib"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_data():
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
    return model, feature_cols, last_cycle, X


def global_importance_plot(explainer, shap_values, X):
    """Which features matter most, across all engines."""
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    out_path = REPORTS_DIR / "shap_summary_global.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved global SHAP summary plot to {out_path}")


def single_engine_explanation(explainer, shap_values, X, last_cycle, unit_number, feature_cols):
    """Why did the model predict THIS RUL for THIS specific engine?"""
    row_idx = last_cycle.index[last_cycle["unit_number"] == unit_number]
    if len(row_idx) == 0:
        print(f"Engine {unit_number} not found in test set.")
        return
    row_idx = row_idx[0]

    plt.figure()
    shap.plots.waterfall(shap_values[row_idx], max_display=12, show=False)
    out_path = REPORTS_DIR / f"shap_engine_{unit_number}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved explanation for engine {unit_number} to {out_path}")

    # Also print the top contributing features as text
    row_shap = shap_values[row_idx]
    contributions = pd.Series(row_shap.values, index=feature_cols).sort_values(
        key=abs, ascending=False
    )
    print(f"\nTop factors for engine {unit_number} "
          f"(predicted RUL = {row_shap.base_values + row_shap.values.sum():.1f}):")
    print(contributions.head(8))


def main():
    model, feature_cols, last_cycle, X = load_model_and_data()

    print("Computing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # 1. Global importance — which sensors matter most overall
    global_importance_plot(explainer, shap_values, X)

    # 2. Per-engine explanation — pick a few interesting engines
    #    (lowest predicted RUL = most at-risk, worth explaining first)
    last_cycle = last_cycle.copy()
    last_cycle["predicted_RUL"] = model.predict(X)
    most_at_risk = last_cycle.nsmallest(3, "predicted_RUL")["unit_number"].tolist()

    print(f"\nExplaining the 3 most at-risk engines: {most_at_risk}")
    for unit in most_at_risk:
        single_engine_explanation(explainer, shap_values, X, last_cycle, unit, feature_cols)


if __name__ == "__main__":
    main()