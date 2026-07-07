from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.common import topk_set_f1, compute_ci95


def evaluate_classification_topk(
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

    drop_cols = [dataset_col, "task_kind", "task_id", "dataset_id", "target"] + hpi_cols
    meta_cols = [c for c in df.columns if c not in drop_cols]

    X = (
        df[meta_cols]
        .replace([np.inf, -np.inf], np.nan)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(-1)
    )

    H_true = df[hpi_cols].to_numpy(dtype=float)

    rows = []
    dataset_rows = []

    for seed in seeds:
        pred_probs = np.zeros_like(H_true, dtype=float)

        for j, hp in enumerate(hpi_cols):
            model_path = model_root / f"seed_{seed}" / "final_models" / hp / "rf_pipeline.joblib"

            if not model_path.exists():
                raise FileNotFoundError(f"Missing classifier model: {model_path}")

            pipe = joblib.load(model_path)
            pred_probs[:, j] = pipe.predict_proba(X)[:, 1]

        f1s = np.array([
            topk_set_f1(pred_probs[i], H_true[i], topk)
            for i in range(len(df))
        ])

        rows.append({
            "benchmark": benchmark_name,
            "approach": "classification",
            "seed": seed,
            f"top{topk}_f1_mean": float(np.mean(f1s)),
            f"top{topk}_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        })

        for i in range(len(df)):
            dataset_rows.append({
                "benchmark": benchmark_name,
                "approach": "classification",
                "seed": seed,
                "row_index": i,
                dataset_col: df.iloc[i][dataset_col] if dataset_col in df.columns else i,
                f"top{topk}_f1": float(f1s[i]),
            })

    per_seed_df = pd.DataFrame(rows)
    per_dataset_df = pd.DataFrame(dataset_rows)

    mean, std, ci95 = compute_ci95(per_seed_df[f"top{topk}_f1_mean"].values)

    summary_df = pd.DataFrame([{
        "benchmark": benchmark_name,
        "approach": "classification",
        "metric": f"top{topk}_f1",
        "mean_over_seeds": mean,
        "std_over_seeds": std,
        "ci95": ci95,
        "mean_pm_ci95": f"{mean:.4f} ± {ci95:.4f}",
    }])

    per_seed_df.to_csv(model_root / f"eval_classification_per_seed_top{topk}.csv", index=False)
    per_dataset_df.to_csv(model_root / f"eval_classification_per_dataset_top{topk}.csv", index=False)
    summary_df.to_csv(model_root / f"eval_classification_summary_top{topk}.csv", index=False)

    return per_seed_df, summary_df, per_dataset_df