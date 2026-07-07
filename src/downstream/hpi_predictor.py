from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.downstream.config import (
    REGRESSION_MODEL_ROOT,
    HOLDOUT_META_CSV,
    SEEDS,
    DATASET_COL,
    HPI_PREFIX,
    HP_NAMES,
)

from src.downstream.utils import normalize_hpi_dict


def load_holdout_meta_features():
    df = pd.read_csv(HOLDOUT_META_CSV)

    if DATASET_COL not in df.columns:
        raise ValueError(f"{HOLDOUT_META_CSV} must contain '{DATASET_COL}'")

    hpi_cols = [c for c in df.columns if c.startswith(HPI_PREFIX)]

    blocked = {
        DATASET_COL,
        "task_kind",
        "task_id",
        "dataset_id",
        "target",
        *hpi_cols,
    }

    meta_cols = [c for c in df.columns if c not in blocked]

    if len(meta_cols) == 0:
        raise ValueError("No meta-feature columns found in holdout meta CSV")

    df_meta = df[[DATASET_COL] + meta_cols].copy()
    df_meta[meta_cols] = (
        df_meta[meta_cols]
        .replace([np.inf, -np.inf], np.nan)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(-1)
    )

    return df_meta, meta_cols


def load_regression_models_for_seed(seed: int):
    seed_dir = REGRESSION_MODEL_ROOT / f"seed_{seed}"
    models = {}

    for hp in HP_NAMES:
        hp_col = f"hpi_{hp}"
        pipe_path = seed_dir / f"pipeline__{hp_col}.joblib"

        if not pipe_path.exists():
            raise FileNotFoundError(f"Missing regression pipeline: {pipe_path}")

        models[hp] = joblib.load(pipe_path)

    return models


def load_all_regression_models():
    return {
        seed: load_regression_models_for_seed(seed)
        for seed in SEEDS
    }


def predict_hpi_for_dataset(
    x_row: pd.DataFrame,
    models_by_seed: dict[int, dict[str, Any]],
):
    seed_preds = []

    for seed in SEEDS:
        preds = {}

        for hp in HP_NAMES:
            yhat = models_by_seed[seed][hp].predict(x_row)[0]
            preds[hp] = float(max(0.0, yhat))

        seed_preds.append(normalize_hpi_dict(preds, HP_NAMES))

    avg_preds = {
        hp: float(np.mean([p[hp] for p in seed_preds]))
        for hp in HP_NAMES
    }

    return normalize_hpi_dict(avg_preds, HP_NAMES)