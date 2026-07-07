import numpy as np
import torch

from pytabkit.models.training.metrics import Metrics


def ensure_binary_probs_shape(probs: np.ndarray) -> np.ndarray:
    """Convert binary probabilities to shape (n_samples, 2)."""
    if probs.ndim == 1:
        return np.vstack([1.0 - probs, probs]).T
    if probs.ndim == 2 and probs.shape[1] == 1:
        return np.hstack([1.0 - probs, probs])
    return probs


def classification_accuracy(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Compute accuracy using PyTabKit's class_error metric."""
    num_classes = len(np.unique(y_true))

    if num_classes == 2:
        probs = ensure_binary_probs_shape(probs)

    logits = np.log(np.clip(probs, 1e-30, 1.0))
    y_pred_t = torch.tensor(logits, dtype=torch.float32)

    if num_classes == 2:
        y_true_t = torch.tensor(y_true, dtype=torch.long).reshape(-1, 1)
    else:
        y_true_t = torch.tensor(y_true, dtype=torch.long)

    class_error = Metrics.apply(y_pred_t, y_true_t, "class_error").item()
    return float(1.0 - class_error)


def regression_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE for regression tasks."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))