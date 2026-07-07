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


def make_output_dir(benchmark_name: str, use_soft_wins: bool):
    wins_name = "soft" if use_soft_wins else "hard"
    return f"smac_optimization/ordering_{benchmark_name}_{wins_name}_5seeds"


def prepare_data(df, dataset_col, task_col, hpi_prefix):
    hpi_cols = [c for c in df.columns if c.startswith(hpi_prefix)]

    drop_cols = [dataset_col] + hpi_cols
    if task_col is not None and task_col in df.columns:
        drop_cols.append(task_col)

    meta_cols = [c for c in df.columns if c not in drop_cols]

    if not hpi_cols:
        raise ValueError("No hpi_ columns found.")
    if not meta_cols:
        raise ValueError("No meta-feature columns found.")

    X_all = (
        df[meta_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(-1)
        .to_numpy(dtype=float)
    )

    H_all = df[hpi_cols].to_numpy(dtype=float)

    return X_all, H_all, meta_cols, hpi_cols