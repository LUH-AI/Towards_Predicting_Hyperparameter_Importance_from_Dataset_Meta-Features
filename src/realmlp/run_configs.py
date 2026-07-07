import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pytabkit.models.alg_interfaces.nn_interfaces import RealMLPParamSampler
from pytabkit.models.sklearn.sklearn_interfaces import (
    RealMLP_TD_Classifier,
    RealMLP_TD_Regressor,
)

from src.data.datasets import (
    infer_task_kind,
    iter_dataset_dirs,
    load_dataset,
)
from src.realmlp.metrics import (
    classification_accuracy,
    regression_mae,
)


HP_NAMES = [
    "lr",
    "num_emb_type",
    "add_front_scale",
    "p_drop",
    "wd",
    "plr_sigma",
    "hidden_sizes",
    "act",
]


def hidden_sizes_to_text(hidden_sizes) -> str:
    """Convert RealMLP hidden size list to compact text."""
    if hidden_sizes == [256, 256, 256]:
        return "256x3"
    if hidden_sizes == [64, 64, 64, 64, 64]:
        return "64x5"
    if hidden_sizes == [512]:
        return "512x1"
    return str(hidden_sizes)


def make_sampler(task_kind: str, hpo_space_name: str):
    """Create RealMLP hyperparameter sampler."""
    return RealMLPParamSampler(
        is_classification=(task_kind == "classification"),
        hpo_space_name=hpo_space_name,
    )


def sample_params(sampler, seed: int, task_kind: str) -> dict[str, Any]:
    """Sample one RealMLP configuration."""
    params = sampler.sample_params(seed=seed)

    # Label smoothing is not used in the shared 8-HP TabArena setup.
    if task_kind == "classification":
        params.pop("ls_eps", None)

    return params


def make_model(task_kind: str, params: dict, seed: int, device: str):
    """Build classifier or regressor."""
    model_cls = RealMLP_TD_Classifier if task_kind == "classification" else RealMLP_TD_Regressor

    try:
        return model_cls(
            device=device,
            verbosity=0,
            random_state=seed,
            **params,
        )
    except TypeError:
        return model_cls(
            device=device,
            verbosity=0,
            **params,
        )


def evaluate_config(
    X_train,
    X_test,
    y_train,
    y_test,
    task_kind: str,
    sampler,
    config_id: int,
    n_epochs: int,
    device: str,
) -> tuple[dict, float, float, float, int, int]:
    """Train and evaluate one sampled RealMLP configuration."""
    params = sample_params(sampler, seed=config_id, task_kind=task_kind)
    params["n_epochs"] = n_epochs

    model = make_model(task_kind, params, seed=config_id, device=device)

    start_time = int(time.time())

    fit_start = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - fit_start

    pred_start = time.time()

    if task_kind == "classification":
        probs = model.predict_proba(X_test)
        metric = classification_accuracy(probs, y_test)
    else:
        y_pred = model.predict(X_test)
        metric = regression_mae(y_test, y_pred)

    predict_seconds = time.time() - pred_start

    end_time = int(time.time())

    return params, metric, fit_seconds, predict_seconds, start_time, end_time

def write_header(path: Path, task_kind: str) -> None:
    """Create trials.csv header."""
    metric_col = (
        "metric:accuracy [0.0; 1.0] (maximize)"
        if task_kind == "classification"
        else "metric:mae [0.0; inf] (minimize)"
    )

    columns = [
        "dataset_name",
        "task_id",
        "split_id",
        "config_id",
        *HP_NAMES,
        metric_col,
        "status",
        "start_time",
        "end_time",
        "fit_seconds",
        "predict_seconds",
        "budget_epochs",
        "seed",
        "additional",
    ]

    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(columns)


def append_row(path: Path, row: list[Any]) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def split_name(split: dict) -> str:
    return f"r{split['repeat']}_f{split['fold']}_s{split['sample']}"


def is_finished(trials_csv: Path, n_configs: int) -> bool:
    """Skip datasets that already contain enough successful runs."""
    if not trials_csv.exists() or trials_csv.stat().st_size == 0:
        return False

    df = pd.read_csv(trials_csv)
    return (df["status"] == "success").sum() >= n_configs


def output_root(task_kind: str, classification_out: Path, regression_out: Path) -> Path:
    return classification_out if task_kind == "classification" else regression_out


def run_realmlp_configs(
    data_root: Path,
    classification_out: Path,
    regression_out: Path,
    n_configs: int = 100,
    n_epochs: int = 30,
    hpo_space_name: str = "default",
    limit_datasets: int | None = None,
    limit_splits: int | None = 1,
    overwrite: bool = False,
    device: str = "cuda:0",
) -> None:
    """Run sampled RealMLP configurations on all exported datasets."""
    classification_out.mkdir(parents=True, exist_ok=True)
    regression_out.mkdir(parents=True, exist_ok=True)

    dataset_dirs = list(iter_dataset_dirs(data_root, limit=limit_datasets))
    print(f"[INFO] Found {len(dataset_dirs)} datasets")

    samplers = {
        "classification": make_sampler("classification", hpo_space_name),
        "regression": make_sampler("regression", hpo_space_name),
    }

    for ds_dir in dataset_dirs:
        meta_path = ds_dir / "meta.json"

        try:
            import json
            meta = json.load(open(meta_path))
            task_kind = infer_task_kind(meta)

            X, y, meta, splits = load_dataset(ds_dir, task_kind)
            dataset_name = meta.get("dataset_name", ds_dir.name)
            task_id = int(meta.get("task_id", -1))

            out_dir = output_root(task_kind, classification_out, regression_out) / ds_dir.name
            trials_csv = out_dir / "trials.csv"

            if trials_csv.exists() and not overwrite and is_finished(trials_csv, n_configs):
                print(f"[SKIP] {ds_dir.name}")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            write_header(trials_csv, task_kind)

            used_splits = splits[:limit_splits] if limit_splits is not None else splits
            sampler = samplers[task_kind]

            print(f"\n[RUN] {ds_dir.name} | {task_kind}")

            for split in used_splits:
                train_idx = np.asarray(split["train_idx"], dtype=int)
                test_idx = np.asarray(split["test_idx"], dtype=int)

                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                split_id = split_name(split)

                for config_id in range(n_configs):
                    try:
                        params, metric, fit_sec, pred_sec, start_time, end_time = evaluate_config(
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            task_kind,
                            sampler,
                            config_id,
                            n_epochs,
                            device,
                        )

                        row = [
                            dataset_name,
                            task_id,
                            split_id,
                            config_id,
                            params.get("lr"),
                            params.get("num_emb_type"),
                            params.get("add_front_scale"),
                            params.get("p_drop"),
                            params.get("wd"),
                            params.get("plr_sigma"),
                            hidden_sizes_to_text(params.get("hidden_sizes")),
                            params.get("act"),
                            metric,
                            "success",
                            start_time,
                            end_time,
                            fit_sec,
                            pred_sec,
                            n_epochs,
                            config_id,
                            "",
                        ]

                    except Exception as e:
                        failed_time = int(time.time())

                        row = [
                            dataset_name,
                            task_id,
                            split_id,
                            config_id,
                            *[""] * len(HP_NAMES),
                            np.nan,
                            "failed",
                            failed_time,
                            failed_time,
                            np.nan,
                            np.nan,
                            n_epochs,
                            config_id,
                            repr(e),
                        ]

                    append_row(trials_csv, row)

                    if (config_id + 1) % 10 == 0:
                        print(f"  config {config_id + 1}/{n_configs}")

            print(f"[DONE] {dataset_name}")

        except Exception as e:
            print(f"[FAIL] {ds_dir.name}: {e}")