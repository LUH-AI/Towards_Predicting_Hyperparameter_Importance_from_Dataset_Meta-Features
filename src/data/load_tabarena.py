import json
from pathlib import Path

import numpy as np
import openml
import pandas as pd


def safe_name(text: str) -> str:
    """Make a dataset name safe for folder names."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def detect_task_kind(task) -> str:
    """Detect whether an OpenML task is classification or regression."""
    task_type_id = getattr(task, "task_type_id", None)

    if task_type_id == 1:
        return "classification"
    if task_type_id == 2:
        return "regression"

    task_type = str(getattr(task, "task_type", "")).lower()
    if "classification" in task_type:
        return "classification"
    if "regression" in task_type:
        return "regression"

    return "unknown"


def save_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def export_task(task_id: int, out_root: Path, suite_id: int) -> None:
    """Download one OpenML task and save data, metadata, and official splits."""
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    task_kind = detect_task_kind(task)

    X, y, categorical_indicator, feature_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute,
    )

    target_name = dataset.default_target_attribute
    df = X.copy()
    df[target_name] = y

    dataset_dir = out_root / f"task_{task_id}__{safe_name(dataset.name)}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(dataset_dir / "data.parquet", index=False)

    categorical_columns = [
        col for col, is_cat in zip(feature_names, categorical_indicator) if is_cat
    ]

    meta = {
        "suite_id": suite_id,
        "task_id": int(task_id),
        "dataset_id": int(dataset.dataset_id),
        "dataset_name": str(dataset.name),
        "target": str(target_name),
        "feature_columns": [str(c) for c in feature_names],
        "categorical_columns": [str(c) for c in categorical_columns],
        "task_kind": task_kind,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_names)),
    }
    save_json(dataset_dir / "meta.json", meta)

    splits = collect_splits(task)
    save_json(dataset_dir / "splits.json", splits)

    print(
        f"Saved {dataset.name}: "
        f"task_kind={task_kind}, rows={len(df)}, splits={len(splits)}"
    )


def collect_splits(task) -> list[dict]:
    """Save all official OpenML train/test splits."""
    repeats, folds, samples = map(int, task.get_split_dimensions())

    splits = []
    for repeat in range(repeats):
        for fold in range(folds):
            for sample in range(samples):
                train_idx, test_idx = task.get_train_test_split_indices(
                    repeat=repeat,
                    fold=fold,
                    sample=sample,
                )
                splits.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "sample": sample,
                        "train_idx": np.asarray(train_idx, dtype=int).tolist(),
                        "test_idx": np.asarray(test_idx, dtype=int).tolist(),
                    }
                )

    return splits


def export_openml_suite(suite_id: int, out_root: Path) -> None:
    """Download all datasets from an OpenML suite."""
    out_root.mkdir(parents=True, exist_ok=True)

    suite = openml.study.get_suite(suite_id)
    print(f"Loaded suite {suite_id} with {len(suite.tasks)} tasks")

    for i, task_id in enumerate(suite.tasks, start=1):
        print(f"\n[{i}/{len(suite.tasks)}] Exporting task_id={task_id}")
        export_task(task_id=task_id, out_root=out_root, suite_id=suite_id)

    print("\nDone")