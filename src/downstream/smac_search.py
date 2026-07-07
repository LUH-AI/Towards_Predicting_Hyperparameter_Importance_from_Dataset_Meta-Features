import time

import numpy as np

from smac import HyperparameterOptimizationFacade, Scenario

from src.downstream.config import N_CONFIGS, ROOT_OUT, HP_NAMES
from src.downstream.search_space import (
    make_configspace,
    make_default_config,
    decode_smac_config,
)
from src.downstream.train_eval import train_one_config


def run_smac_search(
    task_kind: str,
    strategy_name: str,
    dataset_name: str,
    split_id: str,
    active_hps: list[str],
    X_train,
    X_test,
    y_train,
    y_test,
    seed_outer: int,
    device: str
):
    cs = make_configspace(active_hps, seed=seed_outer)

    if task_kind == "classification":
        initial_best = -np.inf
        better = lambda new, best: np.isfinite(new) and new > best
        primary_metric_name = "accuracy"
        to_cost = lambda metric_value: 1.0 - metric_value

    elif task_kind == "regression":
        initial_best = np.inf
        better = lambda new, best: np.isfinite(new) and new < best
        primary_metric_name = "mae"
        to_cost = lambda metric_value: metric_value

    else:
        raise ValueError(f"Unsupported task_kind={task_kind}")

    best_state = {"score": initial_best}
    best_cfg = {"cfg": None}
    best_trial = {"cid": None}

    trials = []
    trial_counter = {"cid": 0}

    default_cfg = make_default_config()
    default_seed = (1000 * int(seed_outer)) % (2**32 - 1)

    (
        metric_dict,
        primary_metric_name_out,
        primary_metric_value,
        fit_seconds,
        predict_seconds,
        fit_start_time,
        fit_end_time,
    ) = train_one_config(
        task_kind=task_kind,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cfg=default_cfg,
        config_seed=default_seed,
        device=device
    )

    best_state["score"] = float(primary_metric_value)
    best_cfg["cfg"] = dict(default_cfg)
    best_trial["cid"] = 0

    trials.append({
        "dataset_name": dataset_name,
        "split_id": split_id,
        "seed": seed_outer,
        "task_kind": task_kind,
        "strategy": strategy_name,
        "active_hps": ", ".join(active_hps),
        "config_id": 0,
        "config_origin": "default",
        "primary_metric_name": primary_metric_name_out,
        "primary_metric_value": float(primary_metric_value),
        "best_primary_metric_so_far": best_state["score"],
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "fit_start_time": fit_start_time,
        "fit_end_time": fit_end_time,
        "config": dict(default_cfg),
        "metrics": metric_dict,
        "status": "success",
    })

    print(
        f"[{dataset_name}][seed={seed_outer}][{strategy_name}] "
        f"config 1/{N_CONFIGS} | origin=default | "
        f"{primary_metric_name_out}={primary_metric_value:.6f} | "
        f"best={best_state['score']:.6f}"
    )

    trial_counter["cid"] = 1

    def objective(config, seed: int = 0):
        cid = trial_counter["cid"]
        trial_counter["cid"] += 1

        cfg = decode_smac_config(config, active_hps)
        config_seed = (1000 * int(seed_outer) + cid) % (2**32 - 1)

        try:
            (
                metric_dict,
                primary_metric_name_out,
                primary_metric_value,
                fit_seconds,
                predict_seconds,
                fit_start_time,
                fit_end_time,
            ) = train_one_config(
                task_kind=task_kind,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                cfg=cfg,
                config_seed=config_seed,
                device=device
            )

            if better(primary_metric_value, best_state["score"]):
                best_state["score"] = float(primary_metric_value)
                best_cfg["cfg"] = dict(cfg)
                best_trial["cid"] = cid

            trials.append({
                "dataset_name": dataset_name,
                "split_id": split_id,
                "seed": seed_outer,
                "task_kind": task_kind,
                "strategy": strategy_name,
                "active_hps": ", ".join(active_hps),
                "config_id": cid,
                "config_origin": "smac",
                "primary_metric_name": primary_metric_name_out,
                "primary_metric_value": float(primary_metric_value),
                "best_primary_metric_so_far": best_state["score"],
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "fit_start_time": fit_start_time,
                "fit_end_time": fit_end_time,
                "config": dict(cfg),
                "metrics": metric_dict,
                "status": "success",
            })

            print(
                f"[{dataset_name}][seed={seed_outer}][{strategy_name}] "
                f"config {cid + 1}/{N_CONFIGS} | origin=smac | "
                f"{primary_metric_name_out}={primary_metric_value:.6f} | "
                f"best={best_state['score']:.6f}"
            )

            return float(to_cost(primary_metric_value))

        except Exception as e:
            trials.append({
                "dataset_name": dataset_name,
                "split_id": split_id,
                "seed": seed_outer,
                "task_kind": task_kind,
                "strategy": strategy_name,
                "active_hps": ", ".join(active_hps),
                "config_id": cid,
                "config_origin": "smac",
                "primary_metric_name": primary_metric_name,
                "primary_metric_value": np.nan,
                "best_primary_metric_so_far": best_state["score"],
                "fit_seconds": np.nan,
                "predict_seconds": np.nan,
                "fit_start_time": int(time.time()),
                "fit_end_time": int(time.time()),
                "config": dict(cfg),
                "metrics": {},
                "status": "failed",
                "error": repr(e),
            })

            print(
                f"[FAIL] [{dataset_name}][seed={seed_outer}][{strategy_name}] "
                f"config {cid} | origin=smac: {e}"
            )

            return 1e10

    scenario = Scenario(
        cs,
        n_trials=N_CONFIGS - 1,
        deterministic=True,
        seed=seed_outer,
        output_directory=ROOT_OUT / "smac_runs" / dataset_name / f"seed_{seed_outer}" / strategy_name,
    )

    smac = HyperparameterOptimizationFacade(
        scenario,
        objective,
        overwrite=True,
    )

    incumbent = smac.optimize()

    result = {
        "dataset_name": dataset_name,
        "split_id": split_id,
        "seed": seed_outer,
        "task_kind": task_kind,
        "strategy": strategy_name,
        "active_hps": list(active_hps),
        "best_trial": best_trial["cid"],
        "best_config": best_cfg["cfg"],
        "best_score": best_state["score"],
        "smac_incumbent": dict(incumbent),
        "trials": trials,
    }

    if task_kind == "classification":
        result["best_accuracy"] = best_state["score"]
    else:
        result["best_mae"] = best_state["score"]

    return result