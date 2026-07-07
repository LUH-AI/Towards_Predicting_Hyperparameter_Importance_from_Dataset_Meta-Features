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


def prepare_columns(df, dataset_col, task_col, hpi_prefix):
    hpi_cols = [c for c in df.columns if c.startswith(hpi_prefix)]

    drop_cols = [dataset_col] + hpi_cols
    if task_col is not None and task_col in df.columns:
        drop_cols.append(task_col)

    meta_cols = [c for c in df.columns if c not in drop_cols]

    if not hpi_cols:
        raise ValueError(f"No HPI columns found with prefix {hpi_prefix}")

    if not meta_cols:
        raise ValueError("No meta-feature columns found.")

    return meta_cols, hpi_cols


def make_X(df, meta_cols):
    return df[meta_cols].replace([np.inf, -np.inf], np.nan).fillna(-1)


def make_binary_target(df, hp, quantile):
    y_raw = df[hp].values.astype(float)
    threshold = np.quantile(y_raw, quantile)
    y = (y_raw >= threshold).astype(int)
    return y, float(threshold)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)