import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def iter_dataset_dirs(data_root: Path, limit: int | None = None) -> Iterable[Path]:
    """Iterate over exported dataset folders."""
    dataset_dirs = [
        p for p in sorted(data_root.iterdir())
        if p.is_dir()
        and (p / "data.parquet").exists()
        and (p / "meta.json").exists()
        and (p / "splits.json").exists()
    ]
    yield from dataset_dirs[:limit] if limit is not None else dataset_dirs


def infer_task_kind(meta: dict) -> str:
    """Read task type from metadata."""
    task_kind = str(meta.get("task_kind", "")).lower().strip()

    if task_kind in {"classification", "regression"}:
        return task_kind

    raise ValueError(f"Unknown task kind in metadata: {task_kind}")


def preprocess_features(df: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, str]:
    """Fill missing values and encode categorical features."""
    target = meta["target"]
    cat_cols = [c for c in meta.get("categorical_columns", []) if c in df.columns]
    feature_cols = [c for c in df.columns if c != target]

    df = df.copy()

    for col in feature_cols:
        if col in cat_cols:
            df[col] = df[col].astype("object").fillna("__MISSING__")
            df[col] = df[col].astype("category").cat.codes.astype(np.int64)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0.0)

    return df, target


def preprocess_target(df: pd.DataFrame, target: str, task_kind: str) -> np.ndarray:
    """Prepare target vector for classification or regression."""
    y = df[target]

    if task_kind == "classification":
        return y.astype("category").cat.codes.to_numpy(dtype=np.int64)

    if task_kind == "regression":
        y = pd.to_numeric(y, errors="coerce")
        y = y.fillna(y.median() if y.notna().any() else 0.0)
        return y.to_numpy(dtype=np.float32)

    raise ValueError(f"Unsupported task_kind: {task_kind}")


def load_dataset(ds_dir: Path, task_kind: str):
    """Load one exported dataset as X, y."""
    meta = read_json(ds_dir / "meta.json")
    df = pd.read_parquet(ds_dir / "data.parquet")

    df, target = preprocess_features(df, meta)
    X = df.drop(columns=[target]).to_numpy()
    y = preprocess_target(df, target, task_kind)

    return X, y, meta, read_json(ds_dir / "splits.json")