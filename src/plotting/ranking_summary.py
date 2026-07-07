import json

import numpy as np
import pandas as pd
from scipy.stats import t


SEEDS = [45, 46, 47, 48, 49]


def ci95(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    return float(t.ppf(0.975, df=n - 1) * values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0


def make_ranking_summary(dataset, mode: str):
    root = dataset[f"ranking_{mode}_root"]

    rows = []

    for seed in SEEDS:
        manifest_path = root / f"seed_{seed}" / "surrogate_hpi_predictor.json"

        if not manifest_path.exists():
            print(f"[SKIP] Missing ranking manifest: {manifest_path}")
            continue

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        fold_top3 = [fold["metrics"]["top3"] for fold in manifest["folds"]]
        fold_spearman = [fold["metrics"]["spearman"] for fold in manifest["folds"]]

        rows.append({
            "seed": seed,
            "top3_mean": float(np.mean(fold_top3)),
            "top3_std_outer": float(np.std(fold_top3, ddof=1)),
            "spearman_mean": float(np.mean(fold_spearman)),
            "spearman_std_outer": float(np.std(fold_spearman, ddof=1)),
        })

    if not rows:
        return

    per_seed_df = pd.DataFrame(rows).sort_values("seed")

    summary_df = pd.DataFrame([
        {
            "metric": "top3",
            "mean_over_5_seeds": float(per_seed_df["top3_mean"].mean()),
            "std_over_5_seeds": float(per_seed_df["top3_mean"].std(ddof=1)),
            "ci95": ci95(per_seed_df["top3_mean"]),
        },
        {
            "metric": "spearman",
            "mean_over_5_seeds": float(per_seed_df["spearman_mean"].mean()),
            "std_over_5_seeds": float(per_seed_df["spearman_mean"].std(ddof=1)),
            "ci95": ci95(per_seed_df["spearman_mean"]),
        },
    ])

    per_seed_df.to_csv(root / "per_seed_top3_spearman.csv", index=False)
    summary_df.to_csv(root / "summary_top3_spearman_95ci.csv", index=False)

    print(f"[SAVED] {root / 'summary_top3_spearman_95ci.csv'}")