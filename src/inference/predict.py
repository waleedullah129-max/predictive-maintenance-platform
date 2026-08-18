"""
predict.py — Load the trained model and generate RUL predictions.
"""
import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_rul_model_latest.joblib"

def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_cols"]

def predict(df: pd.DataFrame) -> pd.Series:
    model, feature_cols = load_model()
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Input data missing required feature columns: {missing}")
    preds = model.predict(df[feature_cols])
    return pd.Series(preds, index=df.index, name="predicted_RUL")

if __name__ == "__main__":
    processed_dir = PROJECT_ROOT / "data" / "processed"
    test = pd.read_parquet(processed_dir / "FD001_test_features.parquet")

    last_cycle = (
        test.sort_values(["unit_number", "time_cycles"])
        .groupby("unit_number")
        .tail(1)
    )
    preds = predict(last_cycle)

    result = last_cycle[["unit_number", "time_cycles"]].copy()
    result["predicted_RUL"] = preds.values
    print(result.head(15))

    out_path = processed_dir / "FD001_predictions.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path}")