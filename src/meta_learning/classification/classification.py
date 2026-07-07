import os
from pathlib import Path
from typing import Optional

import pandas as pd

from src.meta_learning.classification.config import (
    QUANTILE,
    SEEDS,
    N_OUTER_SPLITS,
    N_INNER_SPLITS,
    N_TRIALS,
    HPI_PREFIX,
)

from src.meta_learning.classification.classification_rf import (
    nested_cv_score_rf,
    fit_final_rf_model,
)

from src.meta_learning.classification.utils import (
    ensure_dir,
    load_json,
    save_json,
    prepare_columns,
    make_X,
    make_binary_target,
)


def run_one_seed(
    seed,
    df,
    meta_cols,
    hpi_cols,
    csv_path,
    out_dir,
    benchmark_name,
):
    seed_out_dir = os.path.join(out_dir, f"seed_{seed}")
    ensure_dir(seed_out_dir)

    done_marker = os.path.join(seed_out_dir, "_DONE")
    seed_manifest_path = os.path.join(seed_out_dir, "surrogate_rf_predictor.json")

    if os.path.exists(done_marker) or os.path.exists(seed_manifest_path):
        print(f"[SKIP] {benchmark_name} seed {seed} already finished")

        if os.path.exists(seed_manifest_path):
            return load_json(seed_manifest_path)

        return {
            "type": "rf_hpi_binclass_no_holdout",
            "benchmark": benchmark_name,
            "seed": seed,
            "status": "skipped_existing",
        }

    manifest = {
        "type": "rf_hpi_binclass_no_holdout",
        "benchmark": benchmark_name,
        "seed": seed,
        "quantile": QUANTILE,
        "n_trials": N_TRIALS,
        "outer_splits": N_OUTER_SPLITS,
        "inner_splits": N_INNER_SPLITS,
        "csv_path": str(csv_path),
        "meta_cols": meta_cols,
        "hpi_cols": hpi_cols,
        "results_by_hpi": {},
    }

    rows_for_csv = []

    for hp in hpi_cols:
        print(f"\n=== [{benchmark_name}][seed={seed}] Evaluating {hp} ===")

        X = make_X(df, meta_cols)
        y, threshold = make_binary_target(df, hp, QUANTILE)

        rf_nested = nested_cv_score_rf(
            X=X,
            y=y,
            seed=seed,
            run_tag=hp,
            seed_out_dir=seed_out_dir,
        )

        rf_final = fit_final_rf_model(
            X=X,
            y=y,
            hp_name=hp,
            seed=seed,
            seed_out_dir=seed_out_dir,
        )

        manifest["results_by_hpi"][hp] = {
            "threshold_quantile": QUANTILE,
            "threshold_value": threshold,
            "class_balance": {
                "n_total": int(len(y)),
                "n_pos": int(y.sum()),
                "n_neg": int(len(y) - y.sum()),
            },
            "rf": {
                **rf_nested,
                **rf_final,
            },
        }

        rows_for_csv.append({
            "seed": seed,
            "hpi": hp,
            "threshold_value": threshold,
            "rf_f1_mean": rf_nested["f1_mean"],
            "rf_f1_std": rf_nested["f1_std"],
        })

        print(
            f"[{benchmark_name}][seed={seed}][{hp}] "
            f"RF={rf_nested['f1_mean']:.6f}±{rf_nested['f1_std']:.6f}"
        )

    results_df = pd.DataFrame(rows_for_csv).sort_values(["hpi"])
    results_csv_path = os.path.join(seed_out_dir, "rf_by_hpi.csv")
    results_df.to_csv(results_csv_path, index=False)

    manifest["results_csv_path"] = results_csv_path
    save_json(seed_manifest_path, manifest)

    with open(done_marker, "w", encoding="utf-8") as f:
        f.write("done\n")

    print(f"[{benchmark_name}][seed={seed}] Saved manifest: {seed_manifest_path}")
    print(f"[{benchmark_name}][seed={seed}] Saved CSV: {results_csv_path}")

    return manifest


def run_classification_experiment(
    csv_path: Path,
    out_dir: Path,
    benchmark_name: str,
    dataset_col: str = "dataset_name",
    task_col: Optional[str] = None,
):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(csv_path)
    meta_cols, hpi_cols = prepare_columns(
        df=df,
        dataset_col=dataset_col,
        task_col=task_col,
        hpi_prefix=HPI_PREFIX,
    )

    print("\n" + "=" * 80)
    print(f"Running classification meta-learner for {benchmark_name}")
    print("=" * 80)
    print("CSV:", csv_path)
    print("Output:", out_dir)
    print("Rows:", len(df))
    print("Meta-features:", len(meta_cols))
    print("HPI targets:", len(hpi_cols))
    print("Seeds:", SEEDS)

    all_results = {
        "type": "rf_hpi_binclass_no_holdout_multi_seed",
        "benchmark": benchmark_name,
        "seeds": SEEDS,
        "quantile": QUANTILE,
        "n_trials": N_TRIALS,
        "outer_splits": N_OUTER_SPLITS,
        "inner_splits": N_INNER_SPLITS,
        "csv_path": str(csv_path),
        "meta_cols": meta_cols,
        "hpi_cols": hpi_cols,
        "results_by_seed": {},
    }

    all_rows = []

    for seed in SEEDS:
        print("\n" + "=" * 80)
        print(f"RUNNING {benchmark_name} SEED {seed}")
        print("=" * 80)

        seed_manifest = run_one_seed(
            seed=seed,
            df=df,
            meta_cols=meta_cols,
            hpi_cols=hpi_cols,
            csv_path=csv_path,
            out_dir=str(out_dir),
            benchmark_name=benchmark_name,
        )

        all_results["results_by_seed"][str(seed)] = seed_manifest

        seed_csv = os.path.join(out_dir, f"seed_{seed}", "rf_by_hpi.csv")
        if os.path.exists(seed_csv):
            all_rows.append(pd.read_csv(seed_csv))

    all_manifest_path = os.path.join(out_dir, "surrogate_rf_predictor_all_seeds.json")
    save_json(all_manifest_path, all_results)

    if len(all_rows) > 0:
        combined_df = pd.concat(all_rows, axis=0, ignore_index=True)
        combined_csv_path = os.path.join(out_dir, "rf_by_hpi_all_seeds.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        print("Saved combined CSV:", combined_csv_path)

    print("Saved combined manifest:", all_manifest_path)