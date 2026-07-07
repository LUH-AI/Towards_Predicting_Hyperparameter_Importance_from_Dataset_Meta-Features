import json
import os

import numpy as np
import pandas as pd


def json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=json_default)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def prepare_data(df, dataset_col, task_col, hpi_prefix):
    hpi_cols = [c for c in df.columns if c.startswith(hpi_prefix)]

    non_feature_cols = [
        dataset_col,
        task_col,
        "task_kind",
        "task_id",
        "dataset_id",
        "target",
    ]

    blocked = set(hpi_cols)
    for col in non_feature_cols:
        if col is not None and col in df.columns:
            blocked.add(col)

    feature_cols = [c for c in df.columns if c not in blocked]

    if not hpi_cols:
        raise ValueError(f"No HPI columns found with prefix '{hpi_prefix}'")
    if not feature_cols:
        raise ValueError("No meta-feature columns found.")

    X = (
        df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(-1)
    )

    Y = df[hpi_cols].astype(float).copy()

    return X, Y, feature_cols, hpi_cols