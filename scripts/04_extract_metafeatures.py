from pathlib import Path

from src.metafeatures.build_table import build_metafeature_hpi_tables
from src.metafeatures.build_uci_table import build_uci_metafeature_hpi_tables


EXPERIMENTS = [
    {
        "name": "tabarena",
        "type": "tabarena",
        "data_root": Path("data/raw/tabarena_51"),
        "classification_realmlp_root": Path("results/realmlp/tabarena_classification_8HPs"),
        "regression_realmlp_root": Path("results/realmlp/tabarena_regression_8HPs"),
        "train_csv": Path("data/processed/features_and_importances_tabarena_train.csv"),
        "holdout_csv": Path("data/processed/features_and_importances_tabarena_holdout.csv"),
        "holdout_datasets": [
            "task_363613__Amazon_employee_access",
            "task_363620__Bioresponse",
            "task_363615__Another-Dataset-on-used-Fiat-500",
        ],
    },

    # Uncomment the block below to also build UCI meta-feature tables (requires scripts 02 and 03 to have run for UCI).
    #{
    #    "name": "uci",
    #    "type": "uci",
    #    "tab_bench_data_root": Path("data/raw/uci/tab_bench_data"),
    #    "bin_realmlp_root": Path("results/realmlp/uci_bin_class_9HPs"),
    #    "multi_realmlp_root": Path("results/realmlp/uci_multi_class_9HPs"),
    #    "train_csv": Path("data/processed/features_and_importances_uci_train.csv"),
    #    "holdout_csv": Path("data/processed/features_and_importances_uci_holdout.csv"),
    #    "holdout_datasets": [
    #        "uci_bin_class_9HPs/abalone",
    #        "uci_bin_class_9HPs/adult",
    #        "uci_multi_class_9HPs/abalone",
    #    ],
    #},
]


def main() -> None:
    for exp in EXPERIMENTS:
        print("\n" + "=" * 80)
        print(f"Building table for {exp['name']}")

        if exp["type"] == "tabarena":
            build_metafeature_hpi_tables(
                data_root=exp["data_root"],
                classification_realmlp_root=exp["classification_realmlp_root"],
                regression_realmlp_root=exp["regression_realmlp_root"],
                train_csv=exp["train_csv"],
                holdout_csv=exp["holdout_csv"],
                holdout_dataset_names=exp["holdout_datasets"],
            )

        elif exp["type"] == "uci":
            build_uci_metafeature_hpi_tables(
                tab_bench_data_root=exp["tab_bench_data_root"],
                bin_realmlp_root=exp["bin_realmlp_root"],
                multi_realmlp_root=exp["multi_realmlp_root"],
                train_csv=exp["train_csv"],
                holdout_csv=exp["holdout_csv"],
                holdout_dataset_names=exp["holdout_datasets"],
            )


if __name__ == "__main__":
    main()