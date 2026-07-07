from pathlib import Path

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.evaluation.common import topk_set_f1, compute_ci95


def spearman_rankcorr(pred_scores, true_scores):
    n_hp = len(true_scores)

    rp = np.empty(n_hp)
    rt = np.empty(n_hp)

    rp[np.argsort(-pred_scores)] = np.arange(n_hp)
    rt[np.argsort(-true_scores)] = np.arange(n_hp)

    rho = spearmanr(rp, rt).correlation
    return 0.0 if rho is None or np.isnan(rho) else float(rho)


def evaluate_ranking_topk(
    train_csv_path: Path,
    model_root: Path,
    benchmark_name: str,
    seeds,
    topk: int = 4,
    dataset_col: str = "dataset_name",
    hpi_prefix: str = "hpi_",
):
    df = pd.read_csv(train_csv_path)

    hpi_cols = [c for c in df.columns if c.startswith(hpi_prefix)]
    true_hpi = df[hpi_cols].to_numpy(dtype=float)

    rows = []

    for seed in seeds:
        seed_dir = model_root / f"seed_{seed}"
        manifest_path = seed_dir / "surrogate_hpi_predictor.json"
        wins_path = seed_dir / "wins_pairwise.csv"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        if not wins_path.exists():
            raise FileNotFoundError(f"Missing wins file: {wins_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        wins_df = pd.read_csv(wins_path)
        pred_wins = wins_df[hpi_cols].to_numpy(dtype=float)

        f1s = np.array([
            topk_set_f1(pred_wins[i], true_hpi[i], topk)
            for i in range(len(df))
        ])

        spearman_values = np.array([
            spearman_rankcorr(pred_wins[i], true_hpi[i])
            for i in range(len(df))
        ])

        fold_top3 = [fold["metrics"]["top3"] for fold in manifest["folds"]]
        fold_spearman = [fold["metrics"]["spearman"] for fold in manifest["folds"]]

        rows.append({
            "benchmark": benchmark_name,
            "approach": "ranking",
            "seed": seed,
            "top3_mean": float(np.mean(fold_top3)),
            "spearman_mean": float(np.mean(fold_spearman)),
            f"top{topk}_f1_mean": float(np.mean(f1s)),
            "dataset_spearman_mean": float(np.mean(spearman_values)),
        })

    per_seed_df = pd.DataFrame(rows)

    mean, std, ci95 = compute_ci95(per_seed_df[f"top{topk}_f1_mean"].values)

    summary_df = pd.DataFrame([{
        "benchmark": benchmark_name,
        "approach": "ranking",
        "metric": f"top{topk}_f1",
        "mean_over_seeds": mean,
        "std_over_seeds": std,
        "ci95": ci95,
        "mean_pm_ci95": f"{mean:.4f} ± {ci95:.4f}",
    }])

    per_seed_df.to_csv(model_root / f"eval_ranking_per_seed_top{topk}.csv", index=False)
    summary_df.to_csv(model_root / f"eval_ranking_summary_top{topk}.csv", index=False)

    return per_seed_df, summary_df