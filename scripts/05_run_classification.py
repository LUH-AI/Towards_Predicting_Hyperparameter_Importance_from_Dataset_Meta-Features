from pathlib import Path

from src.meta_learning.classification import run_classification_experiment


DATASETS = [
    {
        "name": "tabarena",
        "train_csv": Path("data/processed/features_and_importances_tabarena_train.csv"),
        "out_dir": Path("smac_optimization/rf_hpi_tabarena_5seeds"),
        "dataset_col": "dataset_name",
        "task_col": None,
    },

    #{
    #   "name": "uci",
    #   "train_csv": Path("data/processed/features_and_importances_uci_train.csv"),
    #    "out_dir": Path("smac_optimization/rf_hpi_uci_5seeds"),
    #    "dataset_col": "dataset_name",
    #    "task_col": None,
    #},
]


def main() -> None:
    for dataset in DATASETS:
        run_classification_experiment(
            csv_path=dataset["train_csv"],
            out_dir=dataset["out_dir"],
            benchmark_name=dataset["name"],
            dataset_col=dataset["dataset_col"],
            task_col=dataset["task_col"],
        )


if __name__ == "__main__":
    main()