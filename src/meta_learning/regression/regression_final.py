import os
from pathlib import Path
from typing import Optional

import pandas as pd

from src.meta_learning.regression.config import (
    SEEDS,
    N_TRIALS,
    N_OUTER_SPLITS,
    N_INNER_SPLITS,
    OPTIMIZE,
    HPI_PREFIX,
)

from src.meta_learning.regression.utils import (
    ensure_dir,
    save_json,
    load_json,
    prepare_data,
)

from src.meta_learning.regression.rf import (
    nested_cv_for_one_hpi,
    fit_final_model,
)


def run_one_seed(
    seed,
    X,
    Y,
    hpi_cols,
    feature_cols,
    train_csv_path,
    out_dir,
    benchmark_name,
):
    seed_out_dir = os.path.join(out_dir, f"seed_{seed}")
    ensure_dir(seed_out_dir)

    done_marker = os.path.join(seed_out_dir, "_DONE")
    seed_manifest_path = os.path.join(seed_out_dir, "surrogate_hpi_predictor.json")

    if os.path.exists(done_marker):
        print(f"[SKIP] {benchmark_name} seed {seed} already finished")

        if os.path.exists(seed_manifest_path):
            finished_manifest = load_json(seed_manifest_path)
            return {
                "manifest_path": seed_manifest_path,
                "models": finished_manifest.get("models", {}),
            }

        return {
            "benchmark": benchmark_name,
            "seed": seed,
            "status": "skipped_done_marker",
        }

    manifest = {
        "type": "per_hp_rf_regression_smac",
        "benchmark": benchmark_name,
        "seed": seed,
        "n_trials": N_TRIALS,
        "outer_splits": N_OUTER_SPLITS,
        "inner_splits": N_INNER_SPLITS,
        "optimized_metric": OPTIMIZE.lower(),
        "train_csv": str(train_csv_path),
        "feature_cols": feature_cols,
        "hpi_cols": hpi_cols,
        "models": {},
    }

    for hp in hpi_cols:
        print(f"\n=== [{benchmark_name}][seed={seed}] NESTED CV for {hp} ===")

        nested_info = nested_cv_for_one_hpi(
            X_df=X,
            y=Y[hp],
            hp_name=hp,
            seed=seed,
            seed_out_dir=seed_out_dir,
        )

        print(
            f"[{benchmark_name}][seed={seed}][{hp}] "
            f"Nested MAE mean±std: "
            f"{nested_info['nested_mae_mean']:.6f} ± "
            f"{nested_info['nested_mae_std']:.6f}"
        )

        final_info = fit_final_model(
            X_df=X,
            y=Y[hp],
            hp_name=hp,
            seed=seed,
            seed_out_dir=seed_out_dir,
        )

        manifest["models"][hp] = {
            **nested_info,
            **final_info,
        }

    save_json(seed_manifest_path, manifest)

    with open(done_marker, "w", encoding="utf-8") as f:
        f.write("done\n")

    print(f"\n[{benchmark_name}][seed={seed}] Saved manifest: {seed_manifest_path}")
    print(f"[{benchmark_name}][seed={seed}] Wrote done marker: {done_marker}")

    return {
        "manifest_path": seed_manifest_path,
        "models": manifest["models"],
    }


def run_regression_experiment(
    train_csv_path: Path,
    out_dir: Path,
    benchmark_name: str,
    dataset_col: str = "dataset_name",
    task_col: Optional[str] = None,
):
    train_csv_path = Path(train_csv_path)
    out_dir = Path(out_dir)

    ensure_dir(out_dir)

    df = pd.read_csv(train_csv_path)

    X, Y, feature_cols, hpi_cols = prepare_data(
        df=df,
        dataset_col=dataset_col,
        task_col=task_col,
        hpi_prefix=HPI_PREFIX,
    )

    print("\n" + "=" * 80)
    print(f"Running regression meta-learner for {benchmark_name}")
    print("=" * 80)
    print("CSV:", train_csv_path)
    print("Output:", out_dir)
    print("Train rows:", len(df))
    print("Meta-features used:", len(feature_cols))
    print("HPI targets:", len(hpi_cols))
    print("Seeds:", SEEDS)

    all_results = {
        "type": "per_hp_rf_regression_smac_multi_seed",
        "benchmark": benchmark_name,
        "seeds": SEEDS,
        "n_trials": N_TRIALS,
        "outer_splits": N_OUTER_SPLITS,
        "inner_splits": N_INNER_SPLITS,
        "optimized_metric": OPTIMIZE.lower(),
        "train_csv": str(train_csv_path),
        "feature_cols": feature_cols,
        "hpi_cols": hpi_cols,
        "results_by_seed": {},
    }

    for seed in SEEDS:
        print("\n" + "=" * 80)
        print(f"RUNNING {benchmark_name} SEED {seed}")
        print("=" * 80)

        seed_result = run_one_seed(
            seed=seed,
            X=X,
            Y=Y,
            hpi_cols=hpi_cols,
            feature_cols=feature_cols,
            train_csv_path=train_csv_path,
            out_dir=str(out_dir),
            benchmark_name=benchmark_name,
        )

        all_results["results_by_seed"][str(seed)] = seed_result

    all_manifest_path = os.path.join(out_dir, "surrogate_hpi_predictor_all_seeds.json")
    save_json(all_manifest_path, all_results)

    print("\nSaved combined manifest:", all_manifest_path)