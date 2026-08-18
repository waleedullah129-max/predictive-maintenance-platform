"""
tune_model.py — Hyperparameter search for the RUL XGBoost model.
Uses the same GroupKFold CV setup as train_model.py so results are
directly comparable to the locked baseline (CV RMSE 16.34 ± 0.73).
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime
from itertools import product
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

RUL_CAP = 125
N_CV_FOLDS = 5

NON_FEATURE_COLS = [
    "unit_number", "time_cycles", "RUL", "RUL_clipped",
    "factory_id", "machine_id", "machine_type",
]

# Small, sensible grid -- keep this modest so it finishes in reasonable time.
# Expand later if you want a finer search.
PARAM_GRID = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05],
    "min_child_weight": [1, 5],
}

BASELINE_CV_RMSE = 16.34  # from the locked README baseline


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def cv_score(df, feature_cols, params, target_col="RUL_clipped", n_splits=N_CV_FOLDS, seed=42):
    gkf = GroupKFold(n_splits=n_splits)
    fold_rmses = []

    for tr_idx, va_idx in gkf.split(df, groups=df["unit_number"]):
        X_tr, y_tr = df.iloc[tr_idx][feature_cols], df.iloc[tr_idx][target_col]
        X_va, y_va = df.iloc[va_idx][feature_cols], df.iloc[va_idx][target_col]

        model = xgb.XGBRegressor(
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            early_stopping_rounds=30,
            eval_metric="rmse",
            **params,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred = model.predict(X_va)
        fold_rmses.append(float(np.sqrt(mean_squared_error(y_va, pred))))

    return float(np.mean(fold_rmses)), float(np.std(fold_rmses))


def main():
    train_feat = pd.read_parquet(PROCESSED_DIR / "FD001_train_features.parquet")
    feature_cols = get_feature_columns(train_feat)
    train_feat["RUL_clipped"] = train_feat["RUL"].clip(upper=RUL_CAP)

    keys = list(PARAM_GRID.keys())
    combos = list(product(*PARAM_GRID.values()))
    print(f"Searching {len(combos)} hyperparameter combinations "
          f"({N_CV_FOLDS}-fold CV each)...\n")

    results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        mean_rmse, std_rmse = cv_score(train_feat, feature_cols, params)
        results.append({**params, "cv_rmse_mean": mean_rmse, "cv_rmse_std": std_rmse})
        print(f"[{i}/{len(combos)}] {params} -> RMSE {mean_rmse:.2f} ± {std_rmse:.2f}")

    results_df = pd.DataFrame(results).sort_values("cv_rmse_mean")
    print("\n=== Top 5 configurations ===")
    print(results_df.head(5).to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest CV RMSE: {best['cv_rmse_mean']:.2f} ± {best['cv_rmse_std']:.2f}")
    print(f"Baseline CV RMSE (locked): {BASELINE_CV_RMSE:.2f}")

    improvement = BASELINE_CV_RMSE - best["cv_rmse_mean"]
    if improvement > 0:
        print(f"-> Improvement: {improvement:.2f} RMSE lower than baseline")
    else:
        print(f"-> No improvement over baseline (worse by {-improvement:.2f})")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = MODEL_DIR / f"tuning_results_{run_id}.json"
    results_df.to_json(out_path, orient="records", indent=2)
    print(f"\nSaved full tuning results to {out_path}")


if __name__ == "__main__":
    main()