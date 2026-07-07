import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TOPK = 4
SUMMARY_CSV = "results/evaluation/top4_f1_summary_all_approaches.csv"

APPROACH_ORDER = ["classification", "ranking", "regression"]
APPROACH_LABELS = ["Classification", "Ranking", "Regression"]


def make_top4_f1_approach_plots(datasets, fig_dir):
    summary_path = pd.Path if False else SUMMARY_CSV
    df = pd.read_csv(summary_path)

    for dataset in datasets:
        name = dataset["name"]
        title = dataset["title"]

        sub = df[df["benchmark"] == name]

        if sub.empty:
            print(f"[SKIP] No Top-4 rows for {name}")
            continue

        means = []
        ci95s = []

        for approach in APPROACH_ORDER:
            row = sub[sub["approach"] == approach]

            if row.empty:
                raise ValueError(f"Missing {approach} Top-4 result for {name}")

            row = row.iloc[0]
            means.append(float(row["mean_over_seeds"]))
            ci95s.append(float(row["ci95"]))

        x = np.arange(len(APPROACH_LABELS))

        plt.figure(figsize=(7, 4.5))
        plt.bar(x, means, yerr=ci95s, capsize=5)
        plt.ylabel(f"Top-{TOPK} set F1", fontsize=14)
        plt.xticks(x, APPROACH_LABELS, fontsize=13)
        plt.yticks(fontsize=13)
        plt.ylim(0, 1)
        plt.title(f"{title}: Top-{TOPK} set F1 across approaches", fontsize=16)
        plt.tight_layout()

        out_path = fig_dir / f"top-{TOPK}-set-{name}_hard.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"[SAVED] {out_path}")