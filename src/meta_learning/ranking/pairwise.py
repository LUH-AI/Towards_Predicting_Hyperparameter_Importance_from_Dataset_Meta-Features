import os
from functools import partial

import joblib
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import top_k_accuracy_score
from scipy.stats import spearmanr

from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
)

from src.meta_learning.ranking.config import (
    N_INNER_SPLITS,
    N_TRIALS,
    K_MIN,
    K_MAX,
    SKIP_TIES,
    OPTIMIZE,
)


def make_configspace(seed):
    cs = ConfigurationSpace(seed=seed)
    cs.add_hyperparameters([
        UniformIntegerHyperparameter("n_estimators", 100, 1500, default_value=600),
        UniformIntegerHyperparameter("max_depth", 2, 50, default_value=20),
        UniformIntegerHyperparameter("min_samples_split", 2, 30, default_value=2),
        UniformIntegerHyperparameter("min_samples_leaf", 1, 30, default_value=1),
        CategoricalHyperparameter("max_features", ["sqrt", "log2", None]),
        CategoricalHyperparameter("bootstrap", [True, False]),
        UniformIntegerHyperparameter("k_best", K_MIN, K_MAX, default_value=K_MIN),
    ])
    return cs


def build_pairwise_dataset(X_ds, H_ds, n_hp, skip_ties=True):
    X_pairs, y_pairs = [], []

    for x_meta, h in zip(X_ds, H_ds):
        for i in range(n_hp):
            ei = np.zeros(n_hp)
            ei[i] = 1.0

            for j in range(n_hp):
                if i == j:
                    continue
                if skip_ties and h[i] == h[j]:
                    continue

                ej = np.zeros(n_hp)
                ej[j] = 1.0

                X_pairs.append(np.concatenate([x_meta, ei, ej]))
                y_pairs.append(1 if h[i] > h[j] else 0)

    X_pairs = np.asarray(X_pairs, dtype=float)
    y_pairs = np.asarray(y_pairs, dtype=int)

    if len(X_pairs) == 0:
        raise ValueError("No pairwise training examples were created.")

    return X_pairs, y_pairs


def make_pair_model(cfg, seed, n_meta, n_hp):
    rf = RandomForestClassifier(
        n_estimators=int(cfg["n_estimators"]),
        max_depth=int(cfg["max_depth"]),
        min_samples_split=int(cfg["min_samples_split"]),
        min_samples_leaf=int(cfg["min_samples_leaf"]),
        max_features=cfg["max_features"],
        bootstrap=bool(cfg["bootstrap"]),
        n_jobs=-1,
        random_state=seed,
    )

    n_pair_features = n_meta + 2 * n_hp
    k = min(int(cfg["k_best"]), n_pair_features)

    mi_func = partial(mutual_info_classif, random_state=seed)

    return Pipeline([
        ("select", SelectKBest(score_func=mi_func, k=k)),
        ("rf", rf),
    ])


def predict_ranking_for_one(pair_model, x_meta, n_hp, use_soft_wins):
    Xq, pairs = [], []

    for i in range(n_hp):
        ei = np.zeros(n_hp)
        ei[i] = 1.0

        for j in range(n_hp):
            if i == j:
                continue

            ej = np.zeros(n_hp)
            ej[j] = 1.0

            Xq.append(np.concatenate([x_meta, ei, ej]))
            pairs.append((i, j))

    Xq = np.asarray(Xq, dtype=float)
    wins = np.zeros(n_hp, dtype=float)

    if use_soft_wins:
        p_win = pair_model.predict_proba(Xq)[:, 1]
        for (i, _), p in zip(pairs, p_win):
            wins[i] += float(p)
    else:
        pred = pair_model.predict(Xq)
        for (i, _), p in zip(pairs, pred):
            if p == 1:
                wins[i] += 1.0

    order = np.argsort(-wins)
    return order, wins


def true_order(h):
    return np.argsort(-h)


def spearman_from_orders(order_pred, order_true, n_hp):
    ranks_pred = np.empty(n_hp, dtype=float)
    ranks_true = np.empty(n_hp, dtype=float)

    ranks_pred[order_pred] = np.arange(n_hp)
    ranks_true[order_true] = np.arange(n_hp)

    rho = spearmanr(ranks_pred, ranks_true).correlation
    return 0.0 if rho is None or np.isnan(rho) else float(rho)


def evaluate_model(model, X_eval, H_eval, n_hp, use_soft_wins):
    y_true = []
    y_score = []
    spears = []

    for x_meta, h_true in zip(X_eval, H_eval):
        order_pred, wins = predict_ranking_for_one(
            pair_model=model,
            x_meta=x_meta,
            n_hp=n_hp,
            use_soft_wins=use_soft_wins,
        )

        order_true = true_order(h_true)

        y_true.append(int(order_true[0]))
        y_score.append(wins.astype(float))
        spears.append(spearman_from_orders(order_pred, order_true, n_hp))

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.vstack(y_score)

    return {
        "top1": float(top_k_accuracy_score(y_true, y_score, k=1, labels=np.arange(n_hp))),
        "top3": float(top_k_accuracy_score(y_true, y_score, k=3, labels=np.arange(n_hp))),
        "spearman": float(np.mean(spears)),
    }


def metric_to_optimize(metrics):
    if OPTIMIZE == "spearman":
        return metrics["spearman"]
    if OPTIMIZE == "top1":
        return metrics["top1"]
    if OPTIMIZE == "top3":
        return metrics["top3"]

    raise ValueError("OPTIMIZE must be 'spearman', 'top1', or 'top3'")


def optimize_pair_model(
    X_train,
    H_train,
    strat_labels,
    n_meta,
    n_hp,
    seed,
    output_directory,
    use_soft_wins,
):
    inner_cv = StratifiedKFold(
        n_splits=N_INNER_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    def objective(cfg, seed: int = 0, **kwargs):
        fold_scores = []

        for fold_j, (tr, va) in enumerate(inner_cv.split(X_train, strat_labels), start=1):
            X_tr, X_va = X_train[tr], X_train[va]
            H_tr, H_va = H_train[tr], H_train[va]

            X_pair_tr, y_pair_tr = build_pairwise_dataset(
                X_tr,
                H_tr,
                n_hp=n_hp,
                skip_ties=SKIP_TIES,
            )

            model_seed = seed + 1000 * fold_j

            model = make_pair_model(
                cfg,
                seed=model_seed,
                n_meta=n_meta,
                n_hp=n_hp,
            )

            model.fit(X_pair_tr, y_pair_tr)

            metrics = evaluate_model(
                model=model,
                X_eval=X_va,
                H_eval=H_va,
                n_hp=n_hp,
                use_soft_wins=use_soft_wins,
            )

            fold_scores.append(metric_to_optimize(metrics))

        return 1.0 - float(np.mean(fold_scores))

    scenario = Scenario(
        make_configspace(seed),
        n_trials=N_TRIALS,
        deterministic=True,
        seed=seed,
        output_directory=output_directory,
    )

    smac = HyperparameterOptimizationFacade(
        scenario,
        objective,
        overwrite=True,
    )

    return dict(smac.optimize())


def fit_pair_model(X_train, H_train, cfg, n_meta, n_hp, seed):
    X_pair, y_pair = build_pairwise_dataset(
        X_train,
        H_train,
        n_hp=n_hp,
        skip_ties=SKIP_TIES,
    )

    model = make_pair_model(
        cfg,
        seed=seed,
        n_meta=n_meta,
        n_hp=n_hp,
    )

    model.fit(X_pair, y_pair)
    return model


def save_final_model(model, path):
    joblib.dump(model, path)


def selected_pair_features(model, meta_cols, hpi_cols):
    selector = model.named_steps["select"]
    support = selector.get_support()

    pair_feature_names = (
        meta_cols
        + [f"left_{hp}" for hp in hpi_cols]
        + [f"right_{hp}" for hp in hpi_cols]
    )

    return np.array(pair_feature_names)[support].tolist()