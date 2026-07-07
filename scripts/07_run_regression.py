from pathlib import Path

from src.meta_learning.regression.regression_final import run_regression_experiment


DATASETS = [
    {
        "name": "tabarena",
        "train_csv": Path("data/processed/features_and_importances_tabarena_train.csv"),
        "out_dir": Path("smac_optimization/hpi_rf_regression_tabarena"),
        "dataset_col": "dataset_name",
        "task_col": None,
    },

    #{
    #    "name": "uci",
    #    "train_csv": Path("data/processed/features_and_importances_uci_train.csv"),
    #    "out_dir": Path("smac_optimization/hpi_rf_regression_uci"),
    #    "dataset_col": "dataset_name",
    #    "task_col": None,
    #},
]


def main() -> None:
    for dataset in DATASETS:
        run_regression_experiment(
            train_csv_path=dataset["train_csv"],
            out_dir=dataset["out_dir"],
            benchmark_name=dataset["name"],
            dataset_col=dataset["dataset_col"],
            task_col=dataset["task_col"],
        )


if __name__ == "__main__":
    main()