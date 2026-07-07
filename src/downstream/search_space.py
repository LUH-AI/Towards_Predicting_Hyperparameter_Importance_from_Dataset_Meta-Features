from typing import Dict

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
)

from src.downstream.config import HP_NAMES, DEFAULT_CONFIG


def make_default_config() -> Dict:
    return {hp: DEFAULT_CONFIG[hp] for hp in HP_NAMES}


def make_configspace(active_hps: list[str], seed: int) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)

    hp_map = {
        "lr": UniformFloatHyperparameter(
            "lr",
            lower=2e-2,
            upper=3e-1,
            default_value=DEFAULT_CONFIG["lr"],
            log=True,
        ),

        "num_emb_type": CategoricalHyperparameter(
            "num_emb_type",
            choices=["none", "pbld", "pl", "plr"],
            default_value=DEFAULT_CONFIG["num_emb_type"],
        ),

        "add_front_scale": CategoricalHyperparameter(
            "add_front_scale",
            choices=[True, False],
            weights=[0.6, 0.4],
            default_value=DEFAULT_CONFIG["add_front_scale"],
        ),

        "p_drop": CategoricalHyperparameter(
            "p_drop",
            choices=[0.0, 0.15, 0.3],
            weights=[0.3, 0.5, 0.2],
            default_value=DEFAULT_CONFIG["p_drop"],
        ),

        "wd": CategoricalHyperparameter(
            "wd",
            choices=[0.0, 0.02],
            default_value=DEFAULT_CONFIG["wd"],
        ),

        "plr_sigma": UniformFloatHyperparameter(
            "plr_sigma",
            lower=0.05,
            upper=0.5,
            default_value=DEFAULT_CONFIG["plr_sigma"],
            log=True,
        ),

        "act": CategoricalHyperparameter(
            "act",
            choices=["relu", "selu", "mish"],
            default_value=DEFAULT_CONFIG["act"],
        ),

        "hidden_sizes": CategoricalHyperparameter(
            "hidden_sizes",
            choices=["256x3", "64x5", "512x1"],
            weights=[0.6, 0.2, 0.2],
            default_value="256x3",
        ),
    }

    for hp in active_hps:
        cs.add_hyperparameter(hp_map[hp])

    return cs


def decode_smac_config(config, active_hps: list[str]) -> Dict:
    cfg = make_default_config()

    for hp in active_hps:
        val = config[hp]

        if hp == "hidden_sizes":
            if val == "256x3":
                val = [256, 256, 256]
            elif val == "64x5":
                val = [64, 64, 64, 64, 64]
            elif val == "512x1":
                val = [512]
            else:
                raise ValueError(f"Unknown hidden_sizes value: {val}")

        cfg[hp] = val

    return cfg