from pathlib import Path


DATA_ROOT = Path("data/raw/tabarena_51")

REGRESSION_MODEL_ROOT = Path("smac_optimization/hpi_rf_regression_tabarena")
HOLDOUT_META_CSV = Path("data/processed/features_and_importances_tabarena_holdout.csv")

ROOT_OUT = Path("hpo_validation_heldout_tabarena")

SEEDS = [45, 46, 47, 48, 49]

N_CONFIGS = 50
N_EPOCHS = 15
TOPK = 4

DATASET_COL = "dataset_name"
HPI_PREFIX = "hpi_"

HP_NAMES = [
    "lr",
    "num_emb_type",
    "add_front_scale",
    "p_drop",
    "wd",
    "plr_sigma",
    "hidden_sizes",
    "act",
]

HOLDOUT_DATASETS = [
    "task_363613__Amazon_employee_access",
    "task_363620__Bioresponse",
    "task_363615__Another-Dataset-on-used-Fiat-500",
]

DEFAULT_CONFIG = {
    "lr": 0.14,
    "num_emb_type": "pbld",
    "add_front_scale": True,
    "p_drop": 0.15,
    "wd": 0.0,
    "plr_sigma": 0.1,
    "hidden_sizes": [256, 256, 256],
    "act": "mish",
}