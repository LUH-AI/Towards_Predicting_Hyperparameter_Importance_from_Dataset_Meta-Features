from pathlib import Path
from src.realmlp.run_configs import run_realmlp_configs
from src.realmlp.run_uci_config import run_uci_realmlp_configs


EXPERIMENTS = [
    {
        "name": "tabarena",
        "type": "tabarena",
        "data_root": Path("data/raw/tabarena_51"),
        "classification_out": Path("results/realmlp/tabarena_classification_8HPs"),
        "regression_out": Path("results/realmlp/tabarena_regression_8HPs"),
    },

    # Uncomment the block below to also run UCI experiments (91 classification datasets, 9 HPs).
    # Requires TAB_BENCH_DATA_DIR to be set; see src/realmlp/run_uci_config.py.
    #{
    #    "name": "uci",
    #    "type": "uci",
    #    "bin_out": Path("results/realmlp/uci_bin_class_9HPs"),
    #    "multi_out": Path("results/realmlp/uci_multi_class_9HPs"),
    #},
]

N_CONFIGS = 100
N_EPOCHS = 30
HPO_SPACE_NAME = "default"

LIMIT_DATASETS = None
LIMIT_SPLITS = 1
OVERWRITE = False
DEVICE = "cuda:0"  # "cpu"


def main() -> None:
    for exp in EXPERIMENTS:
        print("\n" + "=" * 80)
        print(f"Running {exp['name']}")

        if exp["type"] == "tabarena":
            run_realmlp_configs(
                data_root=exp["data_root"],
                classification_out=exp["classification_out"],
                regression_out=exp["regression_out"],
                n_configs=N_CONFIGS,
                n_epochs=N_EPOCHS,
                hpo_space_name=HPO_SPACE_NAME,
                limit_datasets=LIMIT_DATASETS,
                limit_splits=LIMIT_SPLITS,
                overwrite=OVERWRITE,
                device=DEVICE,
            )

        elif exp["type"] == "uci":
            run_uci_realmlp_configs(
                bin_out=exp["bin_out"],
                multi_out=exp["multi_out"],
                n_configs=N_CONFIGS,
                n_epochs=N_EPOCHS,
                hpo_space_name=HPO_SPACE_NAME,
                limit_datasets=LIMIT_DATASETS,
                overwrite=OVERWRITE,
                device=DEVICE,
            )


if __name__ == "__main__":
    main()