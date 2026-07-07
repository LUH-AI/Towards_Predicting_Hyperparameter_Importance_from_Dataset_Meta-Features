import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pymfe.mfe import MFE

from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskDescription

from src.metafeatures.extract import MFE_GROUPS, MFE_SUMMARY


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


def load_normalized_hpi(hpi_path: Path) -> dict:
    """Load normalized mean_total HPI values for UCI 9-HP setup."""
    with open(hpi_path, "r") as f:
        hpi = json.load(f)

    return {f"hpi_{hp}": hpi[hp][2] for hp in HP_NAMES if hp in hpi}


def make_fixed_paths(tab_bench_data_root: Path):
    """Create PyTabKit Paths object with absolute paths."""
    root = tab_bench_data_root.resolve()

    os.environ["TAB_BENCH_DATA_DIR"] = str(root)

    paths = Paths.from_env_variables()

    paths.base_folder = root
    paths.base_path = root
    paths.tasks_path = root / "tasks"
    paths.results_path = root / "results"
    paths.result_summaries_path = root / "result_summaries"
    paths.uci_download_path = root / "uci_download"

    return paths


def iter_uci_tasks(collection_name: str, tab_bench_data_root: Path):
    """Iterate over UCI task descriptions using the local YAML collection."""
    root = tab_bench_data_root.resolve()
    paths = make_fixed_paths(root)

    collection_path = root / "task_collections" / f"{collection_name}.yaml"

    if not collection_path.exists():
        raise FileNotFoundError(f"Missing collection file: {collection_path}")

    with open(collection_path, "r") as f:
        collection = yaml.safe_load(f)

    task_descs = collection["task_descs"]

    print(f"[INFO] Found {len(task_descs)} datasets for {collection_name}")

    for td in task_descs:
        yield TaskDescription(
            task_name=td["task_name"],
            task_source=td["task_source"],
        ), paths


def load_task_arrays(task_desc, paths):
    """Load UCI task arrays from PyTabKit."""
    task_info = task_desc.load_info(paths)
    task = task_info.load_task(paths)
    ds = task.ds

    X_cont = ds.tensors["x_cont"].numpy()
    X_cat = ds.tensors["x_cat"].numpy()
    y = ds.tensors["y"].numpy().astype(int)

    X = np.hstack([X_cont, X_cat])

    return X, y


def preprocess_arrays_for_mfe(X: np.ndarray, y: np.ndarray):
    """Prepare UCI arrays for PyMFE."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    X_df = pd.DataFrame(X)

    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
        X_df[col] = X_df[col].fillna(
            X_df[col].median() if X_df[col].notna().any() else 0.0
        )

    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(y).astype("category").cat.codes.to_numpy()

    return X_df.to_numpy(), y


def extract_metafeatures_from_arrays(X: np.ndarray, y: np.ndarray) -> dict:
    """Extract PyMFE meta-features from arrays."""
    X, y = preprocess_arrays_for_mfe(X, y)

    mfe = MFE(groups=MFE_GROUPS, summary=MFE_SUMMARY)
    mfe.fit(X, y)

    names, values = mfe.extract()

    return dict(zip(names, values))


def make_uci_dataset_row(
    task_desc,
    paths,
    collection_name: str,
    realmlp_root: Path,
) -> dict | None:
    """Create one UCI row with dataset info, meta-features, and normalized HPI."""
    dataset_name = task_desc.task_name
    result_source = realmlp_root.name
    dataset_id = f"{result_source}/{dataset_name}"

    hpi_path = realmlp_root / dataset_name / "hpi_normalized.json"

    if not hpi_path.exists():
        print(f"[SKIP] {dataset_id}: missing hpi_normalized.json", flush=True)
        return None

    row = {
        "dataset_name": dataset_name,
        "task_source": collection_name,
        "dataset_id": dataset_id,
    }

    start = time.time()
    print(f"[META START] {dataset_id}", flush=True)

    X, y = load_task_arrays(task_desc, paths)
    metafeatures = extract_metafeatures_from_arrays(X, y)

    elapsed = time.time() - start
    print(f"[META DONE ] {dataset_id} in {elapsed / 60:.2f} min", flush=True)

    row.update(metafeatures)

    start = time.time()
    print(f"[HPI START ] {dataset_id}", flush=True)

    hpi = load_normalized_hpi(hpi_path)

    elapsed = time.time() - start
    print(f"[HPI DONE  ] {dataset_id} in {elapsed:.2f} sec", flush=True)

    row.update(hpi)

    return row


def build_uci_metafeature_hpi_tables(
    tab_bench_data_root: Path,
    bin_realmlp_root: Path,
    multi_realmlp_root: Path,
    train_csv: Path,
    holdout_csv: Path,
    holdout_dataset_names: list[str],
) -> None:
    """Build UCI train and holdout CSVs directly."""
    train_rows = []
    holdout_rows = []

    holdout_set = set(holdout_dataset_names)
    found_holdout = set()

    collections = [
        ("uci-bin-class", bin_realmlp_root),
        ("uci-multi-class", multi_realmlp_root),
    ]

    for collection_name, realmlp_root in collections:
        print("\n" + "=" * 80)
        print(f"Processing {collection_name}")
        print(f"Using HPI root: {realmlp_root}")

        for task_desc, paths in iter_uci_tasks(collection_name, tab_bench_data_root):
            dataset_name = task_desc.task_name
            dataset_id = f"{realmlp_root.name}/{dataset_name}"

            print(f"[PROCESS] {dataset_id}", flush=True)

            try:
                row = make_uci_dataset_row(
                    task_desc=task_desc,
                    paths=paths,
                    collection_name=collection_name,
                    realmlp_root=realmlp_root,
                )

                if row is None:
                    continue

                if dataset_id in holdout_set:
                    holdout_rows.append(row)
                    found_holdout.add(dataset_id)
                else:
                    train_rows.append(row)

            except Exception as e:
                print(f"[FAIL] {dataset_id}: {e}", flush=True)

    train_df = pd.DataFrame(train_rows)
    holdout_df = pd.DataFrame(holdout_rows)

    train_csv.parent.mkdir(parents=True, exist_ok=True)
    holdout_csv.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_csv, index=False)
    holdout_df.to_csv(holdout_csv, index=False)

    print("\nHoldout datasets:")
    for name in holdout_dataset_names:
        status = "found" if name in found_holdout else "missing"
        print(f"- {name} [{status}]")

    print(f"\nSaved train CSV:   {train_csv}   shape={train_df.shape}")
    print(f"Saved holdout CSV: {holdout_csv} shape={holdout_df.shape}")