"""
train_model.py - Trains the RUL XGBoost model with GroupKFold CV, and now
logs every run (params + metrics + model artifact) to MLflow for
experiment tracking and comparison.
"""

import json
import joblib
import mlflow
import mlflow.xgboost
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

MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT.as_posix()}/mlflow.db")
mlflow.set_experiment("predictive_maintenance_rul")

RUL_CAP = 125
N_CV_FOLDS = 5

NON_FEATURE_COLS = [
    "unit_number", "time_cycles", "RUL", "RUL_clipped",
    "factory_id", "machine_id", "machine_type",
]

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
    eval_metric="rmse",
)


def nasa_score(y_true, y_pred):
    d = np.asarray(y_pred) - np.asarray(y_true)
    return np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))


def load_data():
    train = pd.read_parquet(PROCESSED_DIR / "FD001_train_features.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "FD001_test_features.parquet")
    return train, test


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def split_by_engine(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["unit_number"]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def last_cycle_per_engine(df: pd.DataFrame) -> pd.DataFrame:
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
    print(f"\nCV RMSE: {cv_summary['cv_rmse_mean']:.2f} +/- {cv_summary['cv_rmse_std']:.2f}")
    print(f"CV MAE:  {cv_summary['cv_mae_mean']:.2f} +/- {cv_summary['cv_mae_std']:.2f}")
    return cv_summary


def save_run_artifacts(model, feature_cols, metrics: dict):
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
    feature_cols = get_feature_columns(train_feat)
    print(f"Using {len(feature_cols)} features")

    train_feat["RUL_clipped"] = train_feat["RUL"].clip(upper=RUL_CAP)
    test_feat["RUL_clipped"] = test_feat["RUL"].clip(upper=RUL_CAP)

    with mlflow.start_run(run_name=f"xgb_rul_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params(XGB_PARAMS)
        mlflow.log_param("rul_cap", RUL_CAP)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_cv_folds", N_CV_FOLDS)

        cv_summary = cross_validate(train_feat, feature_cols)
        mlflow.log_metrics({
            "cv_rmse_mean": cv_summary["cv_rmse_mean"],
            "cv_rmse_std": cv_summary["cv_rmse_std"],
            "cv_mae_mean": cv_summary["cv_mae_mean"],
            "cv_mae_std": cv_summary["cv_mae_std"],
        })

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

        val_metrics = evaluate(model, X_val, y_val,
                                "Validation (full trajectory, comparable to CV)")
        mlflow.log_metrics({
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_nasa_score": val_metrics["nasa_score"],
        })

        test_last = last_cycle_per_engine(test_feat)
        test_metrics = evaluate(model, test_last[feature_cols], test_last["RUL_clipped"],
                                 "Test (last cycle per engine -- the real C-MAPSS task)")
        mlflow.log_metrics({
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_nasa_score": test_metrics["nasa_score"],
        })

        importances = pd.Series(model.feature_importances_, index=feature_cols)
        top_features = importances.sort_values(ascending=False).head(10)
        print("\nTop 10 features:")
        print(top_features)

        mlflow.xgboost.log_model(model, artifact_path="model")

        metrics = {
            "run_timestamp": datetime.now().isoformat(),
            "n_features": len(feature_cols),
            "cv": cv_summary,
            "validation_full_trajectory": val_metrics,
            "test_last_cycle": test_metrics,
            "top_10_features": top_features.to_dict(),
            "xgb_params": {k: v for k, v in XGB_PARAMS.items()},
        }
        save_run_artifacts(model, feature_cols, metrics)

        print(f"\nMLflow run logged. View with: mlflow ui --backend-store-uri file:///{MLFLOW_TRACKING_DIR.as_posix()}")


if __name__ == "__main__":
    main()

