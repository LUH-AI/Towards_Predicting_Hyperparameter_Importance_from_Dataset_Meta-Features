import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t


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


def make_classification_f1_plot(dataset, fig_dir):
    root = dataset["classification_root"]
    csv_path = root / "rf_by_hpi_all_seeds.csv"

    if not csv_path.exists():
        print(f"[SKIP] Missing classification CSV: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df["hp_short"] = df["hpi"].str.replace("hpi_", "", regex=False)

    hp_order = [hp for hp in HP_ORDER if hp in df["hp_short"].unique()]
    df["hp_short"] = pd.Categorical(df["hp_short"], categories=hp_order, ordered=True)

    summary = (
        df.groupby("hp_short", observed=False)["rf_f1_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("hp_short")
    )

    summary["ci95"] = summary.apply(
        lambda row: t.ppf(0.975, df=int(row["count"]) - 1)
        * row["std"] / np.sqrt(row["count"])
        if row["count"] > 1 else 0.0,
        axis=1,
    )

    summary["label"] = summary["hp_short"].map(HP_LABEL_MAP)

    summary.to_csv(root / "classification_f1_score_summary_95ci.csv", index=False)

    plt.figure(figsize=(14, 7))
    plt.bar(summary["label"], summary["mean"], yerr=summary["ci95"], capsize=6)
    plt.xlabel("Hyperparameter", fontsize=25)
    plt.ylabel("F1 score", fontsize=25)
    plt.title(f"{dataset['title']}: Classification score per hyperparameter", fontsize=27)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 1)
    plt.tight_layout()

    out_plot = fig_dir / f"{dataset['name']}_classification_f1_per_hp.png"
    plt.savefig(out_plot, dpi=300)
    plt.close()

    print(f"[SAVED] {out_plot}")