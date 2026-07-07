import numpy as np
from scipy.stats import t
from sklearn.metrics import f1_score


def topk_vec(scores, k):
    y = np.zeros(len(scores), dtype=int)
    y[np.argsort(scores)[::-1][:k]] = 1
    return y


def topk_set_f1(pred_scores, true_scores, k):
    return float(
        f1_score(
            topk_vec(true_scores, k),
            topk_vec(pred_scores, k),
            zero_division=0,
        )
    )


def compute_ci95(values):
    values = np.asarray(values, dtype=float)
    n = len(values)

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    ci95 = float(t.ppf(0.975, df=n - 1) * sem) if n > 1 else 0.0

    return mean, std, ci95