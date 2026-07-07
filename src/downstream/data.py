from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.downstream.utils import read_json


def preprocess_dataframe(df: pd.DataFrame, meta: Dict) -> Tuple[np.ndarray, np.ndarray]:
    target_col = meta["target"]
    task_kind = meta.get("task_kind", "unknown")

    cat_cols = [c for c in meta.get("categorical_columns", []) if c in df.columns]
    feat_cols = [c for c in df.columns if c != target_col]
    num_cols = [c for c in feat_cols if c not in cat_cols]

    df = df.copy()

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        median_val = df[c].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[c] = df[c].fillna(median_val)

    for c in cat_cols:
        df[c] = df[c].astype("object").fillna("__MISSING__")
        df[c] = df[c].astype("category").cat.codes.astype(np.int64)

    target_series = df[target_col]

    if task_kind == "classification":
        if not pd.api.types.is_integer_dtype(target_series):
            df[target_col] = target_series.astype("category").cat.codes
        y = df[target_col].to_numpy().astype(int)

    elif task_kind == "regression":
        df[target_col] = pd.to_numeric(target_series, errors="coerce")
        df[target_col] = df[target_col].fillna(df[target_col].median())
        y = df[target_col].to_numpy().astype(float)

    else:
        raise ValueError(f"Unsupported task_kind={task_kind}")

    X = df[feat_cols].to_numpy()
    return X, y


def load_dataset_meta_and_splits(ds_dir: Path):
    meta = read_json(ds_dir / "meta.json")
    splits = read_json(ds_dir / "splits.json")
    return meta, splits


def load_dataset_X_y(ds_dir: Path, meta: Dict):
    df = pd.read_parquet(ds_dir / "data.parquet")
    return preprocess_dataframe(df, meta)


def split_id_from_record(s: Dict) -> str:
    return f"r{s['repeat']}_f{s['fold']}_s{s['sample']}"