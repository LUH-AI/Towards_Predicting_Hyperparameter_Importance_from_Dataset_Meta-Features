from pathlib import Path

from src.meta_learning.ranking.ranking_final import run_ranking_experiment


#USE_SOFT_WINS = True 
USE_SOFT_WINS = False # HardWin


DATASETS = [
    {
        "name": "tabarena",
        "train_csv": Path("data/processed/features_and_importances_tabarena_train.csv"),
        "dataset_col": "dataset_name",
        "task_col": None,
    },

    #{
    #    "name": "uci",
    #    "train_csv": Path("data/processed/features_and_importances_uci_train.csv"),
    #    "dataset_col": "dataset_name",
    #    "task_col": None,
    #},
]


def main() -> None:
    for dataset in DATASETS:
        run_ranking_experiment(
            train_csv_path=dataset["train_csv"],
            benchmark_name=dataset["name"],
            dataset_col=dataset["dataset_col"],
            task_col=dataset["task_col"],
            use_soft_wins=USE_SOFT_WINS,
        )


if __name__ == "__main__":
    main()