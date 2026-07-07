import numpy as np
import pandas as pd

from src.downstream.config import (
    DATA_ROOT,
    ROOT_OUT,
    SEEDS,
    N_CONFIGS,
    N_EPOCHS,
    TOPK,
    DATASET_COL,
    HP_NAMES,
    HOLDOUT_DATASETS,
    DEFAULT_CONFIG,
)

from src.downstream.data import (
    load_dataset_meta_and_splits,
    load_dataset_X_y,
    split_id_from_record,
)

from src.downstream.hpi_predictor import (
    load_holdout_meta_features,
    load_all_regression_models,
    predict_hpi_for_dataset,
)

from src.downstream.smac_search import run_smac_search
from src.downstream.utils import topk_from_scores, save_json


def add_summary_row(summary_rows, dataset_name, task_kind, seed, split_id, predicted_topk, full_res, topk_res):
    if task_kind == "classification":
        full_best = full_res["best_accuracy"]
        topk_best = topk_res["best_accuracy"]
        gain = topk_best - full_best if np.isfinite(full_best) and np.isfinite(topk_best) else np.nan

        summary_rows.append({
            "dataset_name": dataset_name,
            "task_kind": task_kind,
            "seed": seed,
            "split_id": split_id,
            "predicted_topk": ", ".join(predicted_topk),
            "full_best_accuracy": full_best,
            "topk_best_accuracy": topk_best,
            "accuracy_gain_topk_minus_full": gain,
            "full_best_trial": full_res["best_trial"],
            "topk_best_trial": topk_res["best_trial"],
        })

    else:
        full_best = full_res["best_mae"]
        topk_best = topk_res["best_mae"]
        gain = topk_best - full_best if np.isfinite(full_best) and np.isfinite(topk_best) else np.nan

        summary_rows.append({
            "dataset_name": dataset_name,
            "task_kind": task_kind,
            "seed": seed,
            "split_id": split_id,
            "predicted_topk": ", ".join(predicted_topk),
            "full_best_mae": full_best,
            "topk_best_mae": topk_best,
            "mae_gain_topk_minus_full": gain,
            "full_best_trial": full_res["best_trial"],
            "topk_best_trial": topk_res["best_trial"],
        })


def add_trajectory_rows(trajectory_rows, results):
    for res in results:
        for tr in res["trials"]:
            trajectory_rows.append({
                "dataset_name": tr["dataset_name"],
                "task_kind": tr["task_kind"],
                "split_id": tr["split_id"],
                "seed": tr["seed"],
                "strategy": tr["strategy"],
                "active_hps": tr["active_hps"],
                "config_id": tr["config_id"],
                "config_origin": tr["config_origin"],
                "primary_metric_name": tr["primary_metric_name"],
                "primary_metric_value": tr["primary_metric_value"],
                "best_primary_metric_so_far": tr["best_primary_metric_so_far"],
                "status": tr["status"],
            })


def run_downstream_validation(device: str):
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {DATA_ROOT}")

    ROOT_OUT.mkdir(parents=True, exist_ok=True)

    holdout_meta_df, meta_cols = load_holdout_meta_features()
    holdout_meta_df = holdout_meta_df[
        holdout_meta_df[DATASET_COL].isin(HOLDOUT_DATASETS)
    ].copy()

    models_by_seed = load_all_regression_models()

    results_manifest = {
        "experiment": "heldout_hpo_efficiency_validation_smac_default_space",
        "seeds": SEEDS,
        "n_configs": N_CONFIGS,
        "n_epochs": N_EPOCHS,
        "topk": TOPK,
        "holdout_datasets": HOLDOUT_DATASETS,
        "default_config": DEFAULT_CONFIG,
        "results_by_dataset": {},
    }

    summary_rows = []
    trajectory_rows = []

    for dataset_name in HOLDOUT_DATASETS:
        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_name}")
        print("=" * 100)

        ds_dir = DATA_ROOT / dataset_name

        if not ds_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {ds_dir}")

        meta, splits = load_dataset_meta_and_splits(ds_dir)
        task_kind = meta.get("task_kind", "unknown")

        if task_kind not in {"classification", "regression"}:
            raise ValueError(f"Unsupported task_kind for {dataset_name}: {task_kind}")

        used_split = splits[0]
        split_id = split_id_from_record(used_split)

        X, y = load_dataset_X_y(ds_dir, meta)

        train_idx = np.asarray(used_split["train_idx"], dtype=int)
        test_idx = np.asarray(used_split["test_idx"], dtype=int)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        meta_row = holdout_meta_df[holdout_meta_df[DATASET_COL] == dataset_name]

        if len(meta_row) != 1:
            raise ValueError(f"Expected exactly one meta row for {dataset_name}")

        x_meta = meta_row[meta_cols].copy()

        predicted_hpi = predict_hpi_for_dataset(
            x_row=x_meta,
            models_by_seed=models_by_seed,
        )

        predicted_topk = topk_from_scores(predicted_hpi, TOPK)

        print(f"Task kind: {task_kind}")
        print(f"Predicted Top-{TOPK}: {predicted_topk}")

        dataset_store = {
            "task_kind": task_kind,
            "predicted_hpi": predicted_hpi,
            "predicted_topk": predicted_topk,
            "results_by_seed": {},
        }

        for seed in SEEDS:
            print("\n" + "-" * 100)
            print(f"{dataset_name} | seed={seed}")
            print("-" * 100)

            full_res = run_smac_search(
                task_kind=task_kind,
                strategy_name="full_search",
                dataset_name=dataset_name,
                split_id=split_id,
                active_hps=HP_NAMES,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                seed_outer=seed,
                device=device
            )

            topk_res = run_smac_search(
                task_kind=task_kind,
                strategy_name="predicted_topk",
                dataset_name=dataset_name,
                split_id=split_id,
                active_hps=predicted_topk,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                seed_outer=seed,
                device=device
            )

            dataset_store["results_by_seed"][str(seed)] = {
                "full_search": full_res,
                "predicted_topk": topk_res,
            }

            add_summary_row(
                summary_rows,
                dataset_name,
                task_kind,
                seed,
                split_id,
                predicted_topk,
                full_res,
                topk_res,
            )

            add_trajectory_rows(trajectory_rows, [full_res, topk_res])

        results_manifest["results_by_dataset"][dataset_name] = dataset_store

    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset_name", "seed"])
    traj_df = pd.DataFrame(trajectory_rows).sort_values(
        ["dataset_name", "seed", "strategy", "config_id"]
    )

    summary_path = ROOT_OUT / "heldout_hpo_summary.csv"
    traj_path = ROOT_OUT / "heldout_hpo_trajectories.csv"
    manifest_path = ROOT_OUT / "heldout_hpo_results.json"

    summary_df.to_csv(summary_path, index=False)
    traj_df.to_csv(traj_path, index=False)
    save_json(manifest_path, results_manifest)

    print("\nSaved:")
    print(summary_path)
    print(traj_path)
    print(manifest_path)

    print("\nSummary:")
    print(summary_df)