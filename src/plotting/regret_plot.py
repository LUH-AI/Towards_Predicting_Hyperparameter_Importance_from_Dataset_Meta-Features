import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def clean_dataset_name(name: str) -> str:
    if "__" in name:
        name = name.split("__", 1)[1]
    return name.replace("_", " ").replace("-", " ")


def make_regret_plots(validation_root):
    traj_path = validation_root / "heldout_hpo_trajectories.csv"
    out_dir = validation_root / "plots_regret_per_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not traj_path.exists():
        print(f"[SKIP] Missing trajectory file: {traj_path}")
        return

    df = pd.read_csv(traj_path)
    df = df[df["status"] == "success"].copy()

    rows = []

    for (dataset_name, task_kind, seed), sub in df.groupby(
        ["dataset_name", "task_kind", "seed"]
    ):
        sub = sub.copy()

        if task_kind == "classification":
            oracle_best = sub["best_primary_metric_so_far"].max()
            sub["regret"] = oracle_best - sub["best_primary_metric_so_far"]

        elif task_kind == "regression":
            oracle_best = sub["best_primary_metric_so_far"].min()
            sub["regret"] = sub["best_primary_metric_so_far"] - oracle_best

        else:
            continue

        rows.append(sub)

    if not rows:
        print("[SKIP] No regret rows")
        return

    df_regret = pd.concat(rows, ignore_index=True)
    df_regret["regret"] = df_regret["regret"].clip(lower=0.0)

    title_fs = 25
    label_fs = 23
    tick_fs = 18
    legend_fs = 17

    for dataset_name in sorted(df_regret["dataset_name"].unique()):
        sub = df_regret[df_regret["dataset_name"] == dataset_name].copy()

        summary = (
            sub.groupby(["strategy", "config_id"])["regret"]
            .agg(
                median="median",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
                count="count",
            )
            .reset_index()
        )

        plt.figure(figsize=(10, 6))

        for strategy in ["full_search", "predicted_topk"]:
            s = summary[summary["strategy"] == strategy].sort_values("config_id")

            x = s["config_id"].to_numpy() + 1
            y = s["median"].to_numpy()
            q25 = s["q25"].to_numpy()
            q75 = s["q75"].to_numpy()

            label = "Full search" if strategy == "full_search" else "Predicted Top-4"

            plt.step(x, y, where="post", linewidth=3, label=label)
            plt.fill_between(x, q25, q75, step="post", alpha=0.2)

        clean_name = clean_dataset_name(dataset_name)

        plt.xlabel("Trial", fontsize=label_fs)
        plt.ylabel("Regret", fontsize=label_fs)
        plt.title(clean_name, fontsize=title_fs)
        plt.xticks(fontsize=tick_fs)
        plt.yticks(fontsize=tick_fs)
        plt.ylim(bottom=0)
        plt.legend(fontsize=legend_fs)
        plt.tight_layout()

        safe_name = clean_name.lower().replace(" ", "_")
        out_path = out_dir / f"{safe_name}_regret_vs_trials.png"

        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"[SAVED] {out_path}")