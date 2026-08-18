"""
C-MAPSS Turbofan Engine Dataset Loader
Reads train/test/RUL files and returns clean, labeled DataFrames.
"""

import pandas as pd
import numpy as np
from pathlib import Path

INDEX_COLS = ["unit_number", "time_cycles"]
SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# This automatically finds your project folder, no matter where you run the script from
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_fd_dataset(data_dir: str, fd_number: str = "FD001") -> dict:
    data_dir = Path(data_dir)

    def _read(path):
        df = pd.read_csv(path, sep=r"\s+", header=None)
        df = df.iloc[:, :26]
        df.columns = ALL_COLS
        return df

    train = _read(data_dir / f"train_{fd_number}.txt")
    test = _read(data_dir / f"test_{fd_number}.txt")
    rul_truth = pd.read_csv(
        data_dir / f"RUL_{fd_number}.txt", header=None, names=["RUL"]
    )

    max_cycles = train.groupby("unit_number")["time_cycles"].transform("max")
    train["RUL"] = max_cycles - train["time_cycles"]

    rul_truth["unit_number"] = rul_truth.index + 1
    test_max_cycle = test.groupby("unit_number")["time_cycles"].max().reset_index()
    test_max_cycle = test_max_cycle.merge(rul_truth, on="unit_number")
    test_max_cycle = test_max_cycle.rename(columns={"time_cycles": "max_cycle"})

    test = test.merge(test_max_cycle[["unit_number", "max_cycle", "RUL"]], on="unit_number")
    test["RUL"] = test["RUL"] + (test["max_cycle"] - test["time_cycles"])
    test = test.drop(columns=["max_cycle"])

    return {"train": train, "test": test, "rul": rul_truth}


def add_machine_metadata(df: pd.DataFrame, fd_number: str, n_factories: int = 5) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(seed=42)
    unit_ids = df["unit_number"].unique()
    factory_map = {u: f"Factory_{rng.integers(1, n_factories + 1):02d}" for u in unit_ids}
    df["factory_id"] = df["unit_number"].map(factory_map)
    df["machine_id"] = fd_number + "_unit_" + df["unit_number"].astype(str)
    df["machine_type"] = "Turbofan_Engine"
    return df


if __name__ == "__main__":
    data = load_fd_dataset(PROJECT_ROOT / "data" / "raw", "FD001")
    train, test, rul = data["train"], data["test"], data["rul"]

    train = add_machine_metadata(train, "FD001")
    test = add_machine_metadata(test, "FD001")

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print(train[["unit_number", "time_cycles", "RUL", "factory_id"]].head())
    print("\nEngines in train:", train["unit_number"].nunique())
    print("RUL range in train:", train["RUL"].min(), "-", train["RUL"].max())

    train.to_parquet(PROJECT_ROOT / "data" / "processed" / "FD001_train_labeled.parquet", index=False)
    test.to_parquet(PROJECT_ROOT / "data" / "processed" / "FD001_test_labeled.parquet", index=False)
    print("\nSaved labeled parquet files to data/processed/")