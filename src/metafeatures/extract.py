import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymfe.mfe import MFE


MFE_GROUPS = [
    "general",
    "statistical",
    "info-theory",
    "model-based",
    "landmarking",
]

MFE_SUMMARY = ["mean", "sd", "min", "max", "quantiles"]


def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def preprocess_for_mfe(df: pd.DataFrame, meta: dict):
    """Prepare data for PyMFE: impute numeric features and encode categoricals."""
    target = meta["target"]
    cat_cols = [c for c in meta.get("categorical_columns", []) if c in df.columns]
    feature_cols = [c for c in df.columns if c != target]

    df = df.copy()

    for col in feature_cols:
        if col in cat_cols:
            df[col] = df[col].astype("object").fillna("__MISSING__")
            df[col] = df[col].astype("category").cat.codes
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0.0)

    y = df[target]
    if not pd.api.types.is_numeric_dtype(y):
        df[target] = y.astype("category").cat.codes

    X = df[feature_cols].to_numpy()
    y = df[target].to_numpy()

    return X, y


def extract_metafeatures(data_path: Path, meta_path: Path) -> dict:
    """Extract PyMFE meta-features for one dataset."""
    df = pd.read_parquet(data_path)
    meta = read_json(meta_path)

    X, y = preprocess_for_mfe(df, meta)

    mfe = MFE(groups=MFE_GROUPS, summary=MFE_SUMMARY)
    mfe.fit(X, y)

    names, values = mfe.extract()

    return dict(zip(names, values))