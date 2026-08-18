"""
Model Training — Remaining Useful Life (RUL) Regression
=========================================================
Trains an XGBoost regressor on engineered features from build_features.py.

Pipeline:
  1. Load feature-engineered train/test parquet files
  2. Clip RUL target (standard C-MAPSS trick)
  3. Run GroupKFold cross-validation (grouped by engine) -> reliable RMSE estimate
  4. Train a final model on an 80/20 engine split
  5. Evaluate val + test using the SAME methodology (last cycle per engine)
     so the two numbers are directly comparable
  6. Save model + feature list + a timestamped metrics.json for every run
"""

import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RUL_CAP = 125  # standard clipping value for C-MAPSS FD001
N_CV_FOLDS = 5

# Columns to exclude from the feature set.
# IMPORTANT: any target-derived column (RUL, RUL_clipped) or identifier
# (unit_number, machine_id, machine_type, factory_id) must be listed here,
# or it will leak into training as a feature.
NON_FEATURE_COLS = [
    "unit_number", "time_cycles", "RUL", "RUL_clipped",
    "factory_id", "machine_id", "machine_type",
]

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
    eval_metric="rmse",
)


def nasa_score(y_true, y_pred):
    """NASA scoring function: penalizes late (over-)predictions much harder
    than early (under-)predictions, since over-predicting RUL is dangerous
    in a real maintenance setting."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))


def load_data():
    train = pd.read_parquet(PROCESSED_DIR / "FD001_train_features.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "FD001_test_features.parquet")
    return train, test


def get_feature_columns(df: pd.DataFrame) -> list:
    """Feature columns are locked in BEFORE any target-derived column
    (like RUL_clipped) is added, so leakage can't sneak in silently."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def split_by_engine(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """GroupShuffleSplit: an engine's rows never appear in both train and val."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["unit_number"]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def last_cycle_per_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Standard C-MAPSS evaluation convention: score only the final observed
    cycle of each engine (the point where RUL prediction actually matters)."""
    return (
        df.sort_values(["unit_number", "time_cycles"])
        .groupby("unit_number")
        .tail(1)
    )


def evaluate(model, X, y, label: str) -> dict:
    pred = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    score = float(nasa_score(y.values, pred))
    print(f"\n--- {label} ---")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | NASA Score: {score:.1f}")
    return {"rmse": rmse, "mae": mae, "nasa_score": score}


def cross_validate(df, feature_cols, target_col="RUL_clipped",
                    n_splits=N_CV_FOLDS, seed=42):
    """K-fold CV grouped by engine. Gives a much more trustworthy estimate
    of true model performance than a single train/val split, because a
    single split on only ~100 engines can be noisy (lucky/unlucky split)."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_rmses, fold_maes = [], []

    print(f"\n=== {n_splits}-Fold Cross-Validation (grouped by engine) ===")
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(df, groups=df["unit_number"])):
        X_tr, y_tr = df.iloc[tr_idx][feature_cols], df.iloc[tr_idx][target_col]
        X_va, y_va = df.iloc[va_idx][feature_cols], df.iloc[va_idx][target_col]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        pred = model.predict(X_va)
        rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
        mae = float(mean_absolute_error(y_va, pred))
        fold_rmses.append(rmse)
        fold_maes.append(mae)
        print(f"  Fold {fold + 1}/{n_splits}: RMSE={rmse:.2f}  MAE={mae:.2f}")

    cv_summary = {
        "cv_rmse_mean": float(np.mean(fold_rmses)),
        "cv_rmse_std": float(np.std(fold_rmses)),
        "cv_mae_mean": float(np.mean(fold_maes)),
        "cv_mae_std": float(np.std(fold_maes)),
    }
    print(f"\nCV RMSE: {cv_summary['cv_rmse_mean']:.2f} ± {cv_summary['cv_rmse_std']:.2f}")
    print(f"CV MAE:  {cv_summary['cv_mae_mean']:.2f} ± {cv_summary['cv_mae_std']:.2f}")
    return cv_summary


def save_run_artifacts(model, feature_cols, metrics: dict):
    """Every run is saved with a timestamp (for history/comparison) AND
    overwrites a 'latest' pointer (for easy loading in predict.py)."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    versioned_path = MODEL_DIR / f"xgb_rul_model_{run_id}.joblib"
    latest_path = MODEL_DIR / "xgb_rul_model_latest.joblib"
    bundle = {"model": model, "feature_cols": feature_cols}
    joblib.dump(bundle, versioned_path)
    joblib.dump(bundle, latest_path)

    metrics_path = MODEL_DIR / f"metrics_{run_id}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {versioned_path}")
    print(f"Saved 'latest' pointer to {latest_path}")
    print(f"Saved metrics to {metrics_path}")


def main():
    train_feat, test_feat = load_data()

    # Lock feature columns BEFORE adding any target-derived column.
    feature_cols = get_feature_columns(train_feat)
    print(f"Using {len(feature_cols)} features")

    # Clip RUL — capped so the model isn't trained to distinguish
    # near-identical "healthy" engines with huge raw RUL values.
    train_feat["RUL_clipped"] = train_feat["RUL"].clip(upper=RUL_CAP)
    test_feat["RUL_clipped"] = test_feat["RUL"].clip(upper=RUL_CAP)

    # --- Step 1: Cross-validation (gives the real expected performance range) ---
    cv_summary = cross_validate(train_feat, feature_cols)

    # --- Step 2: Final model on a single 80/20 engine split ---
    train_split, val_split = split_by_engine(train_feat)
    X_train = train_split[feature_cols]
    y_train = train_split["RUL_clipped"]
    X_val = val_split[feature_cols]
    y_val = val_split["RUL_clipped"]

    print(f"\nTrain rows: {len(X_train)} | Val rows: {len(X_val)}")
    print(f"Train engines: {train_split['unit_number'].nunique()} "
          f"| Val engines: {val_split['unit_number'].nunique()}")

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    # --- Step 3: Evaluate validation and test ---
    # NOTE: these two are intentionally evaluated differently, and that's OK
    # as long as you don't compare them directly:
    #
    # - Validation uses the FULL trajectory of held-out train engines.
    #   Train engines always run to failure, so "last cycle" there would mean
    #   "RUL = 0" for every engine -- an artificially easy question. Full
    #   trajectory (early-life through failure) is the fair comparison to CV.
    #
    # - Test uses "last cycle only" because that IS the real C-MAPSS task:
    #   test engines are cut off BEFORE failure at an unknown point, and
    #   only that final observed cycle has a meaningful RUL label to score.
    val_metrics = evaluate(model, X_val, y_val,
                            "Validation (full trajectory, comparable to CV)")

    test_last = last_cycle_per_engine(test_feat)
    test_metrics = evaluate(model, test_last[feature_cols], test_last["RUL_clipped"],
                             "Test (last cycle per engine -- the real C-MAPSS task)")

    # --- Feature importance ---
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top_features = importances.sort_values(ascending=False).head(10)
    print("\nTop 10 features:")
    print(top_features)

    # --- Step 4: Save everything ---
    metrics = {
        "run_timestamp": datetime.now().isoformat(),
        "n_features": len(feature_cols),
        "cv": cv_summary,
        "validation_last_cycle": val_metrics,
        "test_last_cycle": test_metrics,
        "top_10_features": top_features.to_dict(),
        "xgb_params": {k: v for k, v in XGB_PARAMS.items()},
    }
    save_run_artifacts(model, feature_cols, metrics)


if __name__ == "__main__":
    main()
