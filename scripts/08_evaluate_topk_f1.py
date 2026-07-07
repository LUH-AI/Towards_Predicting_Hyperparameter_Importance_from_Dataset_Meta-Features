from pathlib import Path

import pandas as pd

from src.evaluation.evaluate_classification import evaluate_classification_topk
from src.evaluation.evaluate_ranking import evaluate_ranking_topk
from src.evaluation.evaluate_regression import evaluate_regression_topk


SEEDS = [45, 46, 47, 48, 49]
TOPK = 4



DATASETS = [
    {
        "name": "tabarena",
        "train_csv": Path("data/processed/features_and_importances_tabarena_train.csv"),
        "classification_root": Path("smac_optimization/hpi_rf_classification_tabarena"),
        "ranking_root": Path("smac_optimization/ordering_tabarena_hard_5seeds"),
        "regression_root": Path("smac_optimization/hpi_rf_regression_tabarena"),
    },

    # {
    #     "name": "uci",
    #     "train_csv": Path("data/processed/features_and_importances_uci_train.csv"),
    #     "classification_root": Path("smac_optimization/hpi_rf_classification_uci"),
    #     "ranking_root": Path("smac_optimization/ordering_uci_hard_5seeds"),
    #     "regression_root": Path("smac_optimization/hpi_rf_regression_uci"),
    # },
]


def main() -> None:
    all_summary = []
    all_per_seed = []

    for dataset in DATASETS:
        name = dataset["name"]
        train_csv = dataset["train_csv"]

        print("\n" + "=" * 100)
        print(f"EVALUATING {name}")
        print("=" * 100)

        cls_per_seed, cls_summary, _ = evaluate_classification_topk(
            train_csv_path=train_csv,
            model_root=dataset["classification_root"],
            benchmark_name=name,
            seeds=SEEDS,
            topk=TOPK,
        )
        all_summary.append(cls_summary)
        all_per_seed.append(cls_per_seed)

        rank_per_seed, rank_summary = evaluate_ranking_topk(
            train_csv_path=train_csv,
            model_root=dataset["ranking_root"],
            benchmark_name=name,
            seeds=SEEDS,
            topk=TOPK,
        )
        all_summary.append(rank_summary)
        all_per_seed.append(rank_per_seed)

        reg_per_seed, reg_summary, _ = evaluate_regression_topk(
            train_csv_path=train_csv,
            model_root=dataset["regression_root"],
            benchmark_name=name,
            seeds=SEEDS,
            topk=TOPK,
        )
        all_summary.append(reg_summary)
        all_per_seed.append(reg_per_seed)

    summary_df = pd.concat(all_summary, ignore_index=True)
    per_seed_df = pd.concat(all_per_seed, ignore_index=True, sort=False)

    out_dir = Path("results/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / f"top{TOPK}_f1_summary_all_approaches.csv"
    per_seed_path = out_dir / f"top{TOPK}_f1_per_seed_all_approaches.csv"

    summary_df.to_csv(summary_path, index=False)
    per_seed_df.to_csv(per_seed_path, index=False)

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(summary_df)

    print("\nSaved:")
    print(summary_path)
    print(per_seed_path)


if __name__ == "__main__":
    main()