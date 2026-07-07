import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection
from pytabkit.models.alg_interfaces.nn_interfaces import RealMLPParamSampler
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier

from src.realmlp.metrics import classification_accuracy

# Need to do this before running code for UCI : export TAB_BENCH_DATA_DIR= /data/raw/uci/tab_bench_data
HP_NAMES = [
    "lr",
    "num_emb_type",
    "add_front_scale",
    "p_drop",
    "wd",
    "plr_sigma",
    "hidden_sizes",
    "act",
    "ls_eps",
]

def hidden_sizes_to_text(hidden_sizes) -> str:
    if hidden_sizes == [256, 256, 256]:
        return "256x3"
    if hidden_sizes == [64, 64, 64, 64, 64]:
        return "64x5"
    if hidden_sizes == [512]:
        return "512x1"
    return str(hidden_sizes)


def iter_uci_tasks(collection_name: str):
    print("TAB_BENCH_DATA_DIR =", os.environ.get("TAB_BENCH_DATA_DIR"))

    paths = Paths.from_env_variables()

    print("paths =", paths)

    task_collection = TaskCollection.from_source(collection_name, paths)

    print(collection_name, "num tasks =", len(task_collection.task_descs))

    for task_desc in task_collection.task_descs:
        yield task_desc, paths


def load_task_arrays(task_desc, paths):
    task_info = task_desc.load_info(paths)
    task = task_info.load_task(paths)
    ds = task.ds

    X_cont = ds.tensors["x_cont"].numpy()
    X_cat = ds.tensors["x_cat"].numpy()
    y = ds.tensors["y"].numpy().astype(int)

    X = np.hstack([X_cont, X_cat])
    return X, y


def make_sampler(hpo_space_name: str):
    return RealMLPParamSampler(
        is_classification=True,
        hpo_space_name=hpo_space_name,
    )


def sample_params(sampler, seed: int) -> dict[str, Any]:
    return sampler.sample_params(seed=seed)


def make_model(params: dict, seed: int, device: str):
    try:
        return RealMLP_TD_Classifier(
            device=device,
            verbosity=0,
            random_state=seed,
            **params,
        )
    except TypeError:
        return RealMLP_TD_Classifier(
            device=device,
            verbosity=0,
            **params,
        )


def evaluate_config(
    X_train,
    X_val,
    y_train,
    y_val,
    sampler,
    config_id: int,
    n_epochs: int,
    device: str,
) -> tuple[dict, float, float, float, int, int]:
    params = sample_params(sampler, seed=config_id)
    params["n_epochs"] = n_epochs

    model = make_model(params, seed=config_id, device=device)

    start_time = int(time.time())

    fit_start = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - fit_start

    pred_start = time.time()
    probs = model.predict_proba(X_val)
    metric = classification_accuracy(probs, y_val)
    predict_seconds = time.time() - pred_start

    end_time = int(time.time())

    return params, metric, fit_seconds, predict_seconds, start_time, end_time


def write_header(path: Path) -> None:
    columns = [
        "dataset_name",
        "task_id",
        "split_id",
        "config_id",
        *HP_NAMES,
        "metric:accuracy [0.0; 1.0] (maximize)",
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


def is_finished(trials_csv: Path, n_configs: int) -> bool:
    if not trials_csv.exists() or trials_csv.stat().st_size == 0:
        return False

    df = pd.read_csv(trials_csv)
    return (df["status"] == "success").sum() >= n_configs


def run_single_uci_collection(
    collection_name: str,
    out_root: Path,
    n_configs: int,
    n_epochs: int,
    hpo_space_name: str,
    limit_datasets: int | None,
    overwrite: bool,
    device: str,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    sampler = make_sampler(hpo_space_name)

    tasks = list(iter_uci_tasks(collection_name))
    if limit_datasets is not None:
        tasks = tasks[:limit_datasets]

    print(f"[INFO] Found {len(tasks)} datasets for {collection_name}")

    for task_desc, paths in tasks:
        dataset_name = task_desc.task_name
        task_id = getattr(task_desc, "task_id", -1)

        out_dir = out_root / dataset_name
        trials_csv = out_dir / "trials.csv"

        if trials_csv.exists() and not overwrite and is_finished(trials_csv, n_configs):
            print(f"[SKIP] {dataset_name}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        write_header(trials_csv)

        print(f"\n[RUN] {dataset_name} | {collection_name}")

        try:
            X, y = load_task_arrays(task_desc, paths)

            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=0,
                stratify=y,
            )

            split_id = "random_state_0_test_0.2"

            for config_id in range(n_configs):
                try:
                    params, metric, fit_sec, pred_sec, start_time, end_time = evaluate_config(
                        X_train,
                        X_val,
                        y_train,
                        y_val,
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
                        params.get("ls_eps"),
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
            print(f"[FAIL] {dataset_name}: {e}")


def run_uci_realmlp_configs(
    bin_out: Path,
    multi_out: Path,
    n_configs: int = 100,
    n_epochs: int = 30,
    hpo_space_name: str = "default",
    limit_datasets: int | None = None,
    overwrite: bool = False,
    device: str = "cuda:0",
) -> None:
    if "TAB_BENCH_DATA_DIR" not in os.environ:
        raise RuntimeError(
            "TAB_BENCH_DATA_DIR is not set. Set it before running:\n"
            "export TAB_BENCH_DATA_DIR=/path/to/tab_bench_data"
        )

    run_single_uci_collection(
        collection_name="uci-bin-class",
        out_root=bin_out,
        n_configs=n_configs,
        n_epochs=n_epochs,
        hpo_space_name=hpo_space_name,
        limit_datasets=limit_datasets,
        overwrite=overwrite,
        device=device,
    )

    run_single_uci_collection(
        collection_name="uci-multi-class",
        out_root=multi_out,
        n_configs=n_configs,
        n_epochs=n_epochs,
        hpo_space_name=hpo_space_name,
        limit_datasets=limit_datasets,
        overwrite=overwrite,
        device=device,
    )