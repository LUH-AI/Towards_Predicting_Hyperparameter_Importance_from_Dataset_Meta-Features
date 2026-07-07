import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SEEDS = [45, 46, 47, 48, 49]
TOP_PER_HP = 5

HP_LABEL_MAP = {
    "hpi_add_front_scale": "Front scaling",
    "hpi_ls_eps": "Label smoothing",
    "hpi_wd": "Weight decay",
    "hpi_p_drop": "Dropout probability",
    "hpi_act": "Activation function",
    "hpi_hidden_sizes": "Hidden layer sizes",
    "hpi_plr_sigma": "PLR sigma",
    "hpi_num_emb_type": "Num. embedding type",
    "hpi_lr": "Learning rate",
}


def wilson_interval(k: int, n: int, z: float = 1.96):
    if n == 0:
        return np.nan, np.nan

    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / denom

    return center - half, center + half


def make_metafeature_heatmap(dataset, fig_dir):
    root = dataset["regression_root"]

    if dataset["name"] != "tabarena":
        print(f"[SKIP] Meta-feature heatmap only for TabArena")
        return

    selection_map = defaultdict(int)
    all_hps = set()
    all_features = set()

    for seed in SEEDS:
        manifest_path = root / f"seed_{seed}" / "surrogate_hpi_predictor.json"

        if not manifest_path.exists():
            print(f"[SKIP] Missing manifest: {manifest_path}")
            return

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for hp, info in manifest["models"].items():
            all_hps.add(hp)

            for feat in info["selected_features"]:
                all_features.add(feat)
                selection_map[(hp, feat)] += 1

    all_hps = sorted(all_hps)
    all_features = sorted(all_features)
    n_seeds = len(SEEDS)

    count_mat = pd.DataFrame(0, index=all_hps, columns=all_features, dtype=int)
    frac_mat = pd.DataFrame(0.0, index=all_hps, columns=all_features, dtype=float)
    ci_low_mat = pd.DataFrame(0.0, index=all_hps, columns=all_features, dtype=float)
    ci_high_mat = pd.DataFrame(0.0, index=all_hps, columns=all_features, dtype=float)

    for hp in all_hps:
        for feat in all_features:
            k = selection_map[(hp, feat)]
            lo, hi = wilson_interval(k, n_seeds)

            count_mat.loc[hp, feat] = k
            frac_mat.loc[hp, feat] = k / n_seeds
            ci_low_mat.loc[hp, feat] = lo
            ci_high_mat.loc[hp, feat] = hi

    selected_cols = set()

    for hp in count_mat.index:
        top_feats = (
            count_mat.loc[hp]
            .sort_values(ascending=False)
            .head(TOP_PER_HP)
            .index
            .tolist()
        )
        selected_cols.update(top_feats)

    selected_cols = (
        count_mat[list(selected_cols)]
        .sum(axis=0)
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    count_plot = count_mat[selected_cols].copy()
    frac_plot = frac_mat[selected_cols].copy()
    ci_low_plot = ci_low_mat[selected_cols].copy()
    ci_high_plot = ci_high_mat[selected_cols].copy()

    count_plot.index = [HP_LABEL_MAP.get(hp, hp) for hp in count_plot.index]
    frac_plot.index = [HP_LABEL_MAP.get(hp, hp) for hp in frac_plot.index]
    ci_low_plot.index = [HP_LABEL_MAP.get(hp, hp) for hp in ci_low_plot.index]
    ci_high_plot.index = [HP_LABEL_MAP.get(hp, hp) for hp in ci_high_plot.index]

    rows = []
    for hp in frac_plot.index:
        for feat in frac_plot.columns:
            rows.append({
                "hp": hp,
                "feature": feat,
                "count_selected": int(count_plot.loc[hp, feat]),
                "n_seeds": n_seeds,
                "fraction_selected": float(frac_plot.loc[hp, feat]),
                "ci95_low": float(ci_low_plot.loc[hp, feat]),
                "ci95_high": float(ci_high_plot.loc[hp, feat]),
            })

    long_df = pd.DataFrame(rows).sort_values(
        ["hp", "count_selected", "feature"],
        ascending=[True, False, True],
    )

    count_plot.to_csv(root / "hp_feature_selection_counts_top5_union.csv")
    frac_plot.to_csv(root / "hp_feature_selection_fraction_top5_union.csv")
    long_df.to_csv(root / "hp_feature_selection_with_ci95_top5_union.csv", index=False)

    fig_w = max(10, 0.45 * len(frac_plot.columns))
    fig_h = max(4, 0.65 * len(frac_plot.index))

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(frac_plot.values, aspect="auto", vmin=0, vmax=1)

    plt.yticks(np.arange(len(frac_plot.index)), frac_plot.index)
    plt.xticks(np.arange(len(frac_plot.columns)), frac_plot.columns, rotation=90)

    for i in range(frac_plot.shape[0]):
        for j in range(frac_plot.shape[1]):
            txt = f"{int(count_plot.iloc[i, j])}/{n_seeds}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=7)

    plt.xlabel("Meta-feature")
    plt.ylabel("Hyperparameter")
    plt.title(f"Selection frequency heatmap (union of top {TOP_PER_HP} features per HP)")

    cbar = plt.colorbar(im)
    cbar.set_label("Fraction selected across 5 seeds")

    plt.tight_layout()

    out_plot = fig_dir / "tabarena_hp_feature_selection_heatmap_top5_union.png"
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {out_plot}")