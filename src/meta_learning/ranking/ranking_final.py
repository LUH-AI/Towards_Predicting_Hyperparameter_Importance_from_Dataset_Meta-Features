import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from src.meta_learning.ranking.config import (
    SEEDS,
    N_OUTER_SPLITS,
    N_INNER_SPLITS,
    N_TRIALS,
    SKIP_TIES,
    OPTIMIZE,
    HPI_PREFIX,
)

from src.meta_learning.ranking.utils import (
    ensure_dir,
    save_json,
    load_json,
    make_output_dir,
    prepare_data,
)

from src.meta_learning.ranking.pairwise import (
    optimize_pair_model,
    fit_pair_model,
    evaluate_model,
    predict_ranking_for_one,
    true_order,
    spearman_from_orders,
    save_final_model,
    selected_pair_features,
)


def run_one_seed(
    seed,
    df,
    X_all,
    H_all,
    hpi_cols,
    meta_cols,
    train_csv_path,
    out_dir,
    benchmark_name,
    dataset_col,
    task_col,
    use_soft_wins,
):
    n_ds, n_meta = X_all.shape
    n_hp = H_all.shape[1]

    seed_out_dir = os.path.join(out_dir, f"seed_{seed}")
    smac_run_dir = os.path.join(seed_out_dir, "smac_runs_pairwise_rf")

    ensure_dir(seed_out_dir)
    ensure_dir(smac_run_dir)

    done_marker = os.path.join(seed_out_dir, "_DONE")
    manifest_path = os.path.join(seed_out_dir, "surrogate_hpi_predictor.json")
    wins_path = os.path.join(seed_out_dir, "wins_pairwise.csv")
    final_model_path = os.path.join(seed_out_dir, "pairwise_rf_pipeline.joblib")

    if os.path.exists(done_marker):
        print(f"[SKIP] {benchmark_name} seed {seed} already finished")

        if os.path.exists(manifest_path):
            return load_json(manifest_path)

        return {
            "approach": "pairwise_ranking_rf_smac",
            "benchmark": benchmark_name,
            "seed": seed,
            "status": "skipped_done_marker",
        }

    strat_labels_all = np.argmax(H_all, axis=1)

    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    oof_wins = np.zeros((n_ds, n_hp), dtype=float)

    manifest = {
        "approach": "pairwise_ranking_rf_smac",
        "benchmark": benchmark_name,
        "seed": seed,
        "outer_splits": N_OUTER_SPLITS,
        "inner_splits": N_INNER_SPLITS,
        "n_trials": N_TRIALS,
        "optimize": OPTIMIZE,
        "use_soft_wins": use_soft_wins,
        "skip_ties": SKIP_TIES,
        "train_csv": str(train_csv_path),
        "hpi_cols": hpi_cols,
        "meta_cols": meta_cols,
        "folds": [],
        "final_model": {},
    }

    for outer_i, (tr_ds, te_ds) in enumerate(outer_cv.split(X_all, strat_labels_all), start=1):
        X_tr, X_te = X_all[tr_ds], X_all[te_ds]
        H_tr, H_te = H_all[tr_ds], H_all[te_ds]
        strat_tr = strat_labels_all[tr_ds]

        fold_seed = seed + outer_i
        fold_outdir = os.path.join(smac_run_dir, f"outer_fold_{outer_i}")

        best_cfg = optimize_pair_model(
            X_train=X_tr,
            H_train=H_tr,
            strat_labels=strat_tr,
            n_meta=n_meta,
            n_hp=n_hp,
            seed=fold_seed,
            output_directory=fold_outdir,
            use_soft_wins=use_soft_wins,
        )

        best_model = fit_pair_model(
            X_train=X_tr,
            H_train=H_tr,
            cfg=best_cfg,
            n_meta=n_meta,
            n_hp=n_hp,
            seed=fold_seed,
        )

        metrics = evaluate_model(
            model=best_model,
            X_eval=X_te,
            H_eval=H_te,
            n_hp=n_hp,
            use_soft_wins=use_soft_wins,
        )

        for idx_global, x_meta in zip(te_ds, X_te):
            _, wins = predict_ranking_for_one(
                pair_model=best_model,
                x_meta=x_meta,
                n_hp=n_hp,
                use_soft_wins=use_soft_wins,
            )
            oof_wins[idx_global] = wins

        print(
            f"[{benchmark_name}][seed={seed}] [Outer fold {outer_i}] "
            f"top1={metrics['top1']:.6f} "
            f"top3={metrics['top3']:.6f} "
            f"spearman={metrics['spearman']:.6f}"
        )

        manifest["folds"].append({
            "outer_fold": outer_i,
            "train_indices": tr_ds.tolist(),
            "test_indices": te_ds.tolist(),
            "best_cfg": best_cfg,
            "smac_dir": fold_outdir,
            "metrics": metrics,
        })

    wins_df = pd.DataFrame(oof_wins, columns=hpi_cols)

    if dataset_col in df.columns:
        wins_df.insert(0, dataset_col, df[dataset_col].values)

    if task_col is not None and task_col in df.columns:
        insert_pos = 1 if dataset_col in wins_df.columns else 0
        wins_df.insert(insert_pos, task_col, df[task_col].values)

    wins_df.to_csv(wins_path, index=False)

    final_smac_dir = os.path.join(smac_run_dir, "final_full_train")

    final_best_cfg = optimize_pair_model(
        X_train=X_all,
        H_train=H_all,
        strat_labels=strat_labels_all,
        n_meta=n_meta,
        n_hp=n_hp,
        seed=seed,
        output_directory=final_smac_dir,
        use_soft_wins=use_soft_wins,
    )

    final_model = fit_pair_model(
        X_train=X_all,
        H_train=H_all,
        cfg=final_best_cfg,
        n_meta=n_meta,
        n_hp=n_hp,
        seed=seed,
    )

    save_final_model(final_model, final_model_path)

    manifest["final_model"] = {
        "best_cfg": final_best_cfg,
        "model_path": final_model_path,
        "selected_pair_features": selected_pair_features(final_model, meta_cols, hpi_cols),
        "smac_dir": final_smac_dir,
    }

    save_json(manifest_path, manifest)

    with open(done_marker, "w", encoding="utf-8") as f:
        f.write("done\n")

    print(f"[{benchmark_name}][seed={seed}] Saved manifest: {manifest_path}")
    print(f"[{benchmark_name}][seed={seed}] Saved wins: {wins_path}")
    print(f"[{benchmark_name}][seed={seed}] Saved final model: {final_model_path}")

    return manifest


def run_ranking_experiment(
    train_csv_path: Path,
    benchmark_name: str,
    dataset_col: str = "dataset_name",
    task_col: Optional[str] = None,
    use_soft_wins: bool = True,
    out_dir: Optional[Path] = None,
):
    train_csv_path = Path(train_csv_path)

    if out_dir is None:
        out_dir = Path(make_output_dir(benchmark_name, use_soft_wins))

    ensure_dir(out_dir)

    df = pd.read_csv(train_csv_path)

    X_all, H_all, meta_cols, hpi_cols = prepare_data(
        df=df,
        dataset_col=dataset_col,
        task_col=task_col,
        hpi_prefix=HPI_PREFIX,
    )

    print("\n" + "=" * 80)
    print(f"Running ranking meta-learner for {benchmark_name}")
    print("=" * 80)
    print("CSV:", train_csv_path)
    print("Output:", out_dir)
    print("Use soft wins:", use_soft_wins)
    print("Train rows:", X_all.shape[0])
    print("Meta-features:", X_all.shape[1])
    print("HPI targets:", H_all.shape[1])
    print("Seeds:", SEEDS)

    all_results = {
        "approach": "pairwise_ranking_rf_smac_multi_seed",
        "benchmark": benchmark_name,
        "seeds": SEEDS,
        "outer_splits": N_OUTER_SPLITS,
        "inner_splits": N_INNER_SPLITS,
        "n_trials": N_TRIALS,
        "optimize": OPTIMIZE,
        "use_soft_wins": use_soft_wins,
        "skip_ties": SKIP_TIES,
        "train_csv": str(train_csv_path),
        "hpi_cols": hpi_cols,
        "meta_cols": meta_cols,
        "results_by_seed": {},
    }

    for seed in SEEDS:
        print("\n" + "=" * 80)
        print(f"RUNNING {benchmark_name} SEED {seed}")
        print("=" * 80)

        seed_manifest = run_one_seed(
            seed=seed,
            df=df,
            X_all=X_all,
            H_all=H_all,
            hpi_cols=hpi_cols,
            meta_cols=meta_cols,
            train_csv_path=train_csv_path,
            out_dir=str(out_dir),
            benchmark_name=benchmark_name,
            dataset_col=dataset_col,
            task_col=task_col,
            use_soft_wins=use_soft_wins,
        )

        all_results["results_by_seed"][str(seed)] = seed_manifest

    all_manifest_path = os.path.join(out_dir, "surrogate_hpi_predictor_all_seeds.json")
    save_json(all_manifest_path, all_results)

    print("\nSaved combined manifest:", all_manifest_path)