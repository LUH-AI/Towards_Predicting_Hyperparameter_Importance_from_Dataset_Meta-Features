import time
from typing import Dict

import numpy as np
import torch

from pytabkit.models.sklearn.sklearn_interfaces import (
    RealMLP_TD_Classifier,
    RealMLP_TD_Regressor,
)
from pytabkit.models.training.metrics import Metrics

from src.downstream.config import N_EPOCHS


def ensure_binary_probs_shape(p: np.ndarray) -> np.ndarray:
    if p.ndim == 1:
        return np.vstack([1.0 - p, p]).T
    if p.ndim == 2 and p.shape[1] == 1:
        return np.hstack([1.0 - p, p])
    return p


def probs_to_logits(probs_np: np.ndarray) -> np.ndarray:
    p = np.clip(probs_np, 1e-30, 1.0)
    return np.log(p)


def get_num_classes(y_train: np.ndarray, y_test: np.ndarray) -> int:
    return int(len(np.unique(np.concatenate([y_train, y_test]))))


def compute_classification_metrics_from_probs(
    probs: np.ndarray,
    y_true: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    metrics_obj = Metrics.defaults([num_classes])
    metric_names = metrics_obj.metric_names

    if num_classes == 2:
        probs = ensure_binary_probs_shape(probs)

    logits_np = probs_to_logits(probs)
    y_pred_t = torch.tensor(logits_np, dtype=torch.float32)

    if num_classes == 2:
        y_true_t = torch.tensor(y_true, dtype=torch.long).reshape(-1, 1)
    else:
        y_true_t = torch.tensor(y_true, dtype=torch.long)

    metric_dict = {}

    for m in metric_names:
        try:
            metric_dict[m] = float(Metrics.apply(y_pred_t, y_true_t, m).item())
        except Exception:
            metric_dict[m] = float("nan")

    class_error = metric_dict.get("class_error", np.nan)
    metric_dict["accuracy"] = (
        float(1.0 - class_error) if np.isfinite(class_error) else float("nan")
    )

    return metric_dict


def compute_regression_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }


def make_model(task_kind: str, hp_full: dict, config_seed: int, device: str):
    if task_kind == "classification":
        model_cls = RealMLP_TD_Classifier
    elif task_kind == "regression":
        model_cls = RealMLP_TD_Regressor
    else:
        raise ValueError(f"Unsupported task_kind={task_kind}")

    try:
        return model_cls(
            device=device,
            verbosity=0,
            random_state=config_seed,
            **hp_full,
        )
    except TypeError:
        return model_cls(
            device=device,
            verbosity=0,
            **hp_full,
        )


def train_one_config(
    task_kind: str,
    X_train,
    X_test,
    y_train,
    y_test,
    cfg: dict,
    config_seed: int,
    device: str
):
    hp_full = dict(cfg)
    hp_full["n_epochs"] = N_EPOCHS

    model = make_model(task_kind, hp_full, config_seed, device=device)

    fit_start = time.time()
    model.fit(X_train, y_train)
    fit_end = time.time()

    pred_start = time.time()

    if task_kind == "classification":
        probs = model.predict_proba(X_test)
        num_classes = get_num_classes(y_train, y_test)
        metric_dict = compute_classification_metrics_from_probs(probs, y_test, num_classes)
        primary_metric_name = "accuracy"
        primary_metric_value = metric_dict.get("accuracy", np.nan)

    else:
        y_pred = model.predict(X_test)
        metric_dict = compute_regression_metrics(y_pred, y_test)
        primary_metric_name = "mae"
        primary_metric_value = metric_dict.get("mae", np.nan)

    pred_end = time.time()

    return (
        metric_dict,
        primary_metric_name,
        primary_metric_value,
        fit_end - fit_start,
        pred_end - pred_start,
        int(fit_start),
        int(fit_end),
    )