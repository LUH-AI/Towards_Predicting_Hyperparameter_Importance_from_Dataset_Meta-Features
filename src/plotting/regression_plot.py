import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t


SEEDS = [45, 46, 47, 48, 49]

HP_ORDER = [
    "lr", "num_emb_type", "add_front_scale", "p_drop",
    "wd", "plr_sigma", "hidden_sizes", "act", "ls_eps",
]

HP_LABEL_MAP = {
    "lr": "Learning rate",
    "num_emb_type": "Num. embedding type",
    "add_front_scale": "Front scaling",
    "p_drop": "Dropout probability",
    "wd": "Weight decay",
    "plr_sigma": "PLR sigma",
    "hidden_sizes": "Hidden layer sizes",
    "act": "Activation function",
    "ls_eps": "Label smoothing",
}


def make_regression_score_plot(dataset, fig_dir):
    root = dataset["regression_root"]

    rows = []

    for seed in SEEDS:
        manifest_path = root / f"seed_{seed}" / "surrogate_hpi_predictor.json"

        if not manifest_path.exists():
            print(f"[SKIP] Missing regression manifest: {manifest_path}")
            continue

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for hp, info in manifest["models"].items():
            hp_short = hp.replace("hpi_", "")
            mae = float(info["nested_mae_mean"])

            rows.append({
                "seed": seed,
                "hpi": hp,
                "hp_short": hp_short,
                "mae_mean": mae,
                "score": 1.0 - mae,
            })

    if not rows:
        return

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["hpi", "hp_short"])["score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["ci95"] = summary.apply(
        lambda row: t.ppf(0.975, df=int(row["count"]) - 1)
        * row["std"] / np.sqrt(row["count"])
        if row["count"] > 1 else 0.0,
        axis=1,
    )

    hp_order = [hp for hp in HP_ORDER if hp in summary["hp_short"].values]
    summary["hp_short"] = pd.Categorical(summary["hp_short"], categories=hp_order, ordered=True)
    summary = summary.sort_values("hp_short")
    summary["label"] = summary["hp_short"].map(HP_LABEL_MAP)

    summary.to_csv(root / "normalized_regression_score_summary_95ci.csv", index=False)
    df.to_csv(root / "normalized_regression_score_per_seed.csv", index=False)

    plt.figure(figsize=(14, 7))
    plt.bar(summary["label"], summary["mean"], yerr=summary["ci95"], capsize=6)
    plt.xlabel("Hyperparameter", fontsize=25)
    plt.ylabel(r"$(1 - MAE)$", fontsize=25)
    plt.title(f"{dataset['title']}: Normalized regression score per hyperparameter", fontsize=27)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 1.05)
    plt.tight_layout()

    out_plot = fig_dir / f"{dataset['name']}_regression_score_per_hp.png"
    plt.savefig(out_plot, dpi=300)
    plt.close()

    print(f"[SAVED] {out_plot}")