import os
from functools import partial

import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
)

from src.meta_learning.classification.config import (
    N_OUTER_SPLITS,
    N_INNER_SPLITS,
    N_TRIALS,
    K_MIN,
    K_MAX,
)


def make_rf(cfg, seed):
    return RandomForestClassifier(
        n_estimators=int(cfg["n_estimators"]),
        max_depth=int(cfg["max_depth"]),
        min_samples_split=int(cfg["min_samples_split"]),
        min_samples_leaf=int(cfg["min_samples_leaf"]),
        max_features=cfg["max_features"],
        bootstrap=bool(cfg["bootstrap"]),
        criterion=str(cfg["criterion"]),
        n_jobs=-1,
        random_state=seed,
    )


def rf_configspace(seed):
    cs = ConfigurationSpace(seed=seed)
    cs.add_hyperparameters([
        UniformIntegerHyperparameter("n_estimators", 100, 1500),
        UniformIntegerHyperparameter("max_depth", 2, 50),
        UniformIntegerHyperparameter("min_samples_split", 2, 30),
        UniformIntegerHyperparameter("min_samples_leaf", 1, 30),
        CategoricalHyperparameter("max_features", ["sqrt", "log2", None]),
        CategoricalHyperparameter("bootstrap", [True, False]),
        CategoricalHyperparameter("criterion", ["gini", "entropy"]),
        UniformIntegerHyperparameter("k_best", K_MIN, K_MAX),
    ])
    return cs


def build_pipeline(cfg, seed, n_features):
    k = min(int(cfg["k_best"]), n_features)
    mi_func = partial(mutual_info_classif, random_state=seed)

    return Pipeline([
        ("select", SelectKBest(score_func=mi_func, k=k)),
        ("clf", make_rf(cfg, seed)),
    ])


def smac_objective(X, y, inner_cv, seed):
    def objective(cfg, seed: int = 0, **kwargs):
        pipe = build_pipeline(cfg, seed=seed, n_features=X.shape[1])

        scores = cross_val_score(
            pipe,
            X,
            y,
            cv=inner_cv,
            scoring="f1",
            n_jobs=-1,
            error_score="raise",
        )

        return 1.0 - float(scores.mean())

    return objective


def nested_cv_score_rf(X, y, seed, run_tag, seed_out_dir):
    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    outer_scores = []
    best_cfgs_by_fold = []

    for i, (tr, va) in enumerate(outer_cv.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]

        fold_seed = seed + i

        inner_cv = StratifiedKFold(
            n_splits=N_INNER_SPLITS,
            shuffle=True,
            random_state=fold_seed,
        )

        scenario = Scenario(
            rf_configspace(fold_seed),
            n_trials=N_TRIALS,
            deterministic=True,
            seed=fold_seed,
            output_directory=os.path.join(
                seed_out_dir,
                "smac_runs",
                run_tag,
                "rf",
                f"fold_{i}",
            ),
        )

        smac = HyperparameterOptimizationFacade(
            scenario,
            smac_objective(X_tr, y_tr, inner_cv, fold_seed),
            overwrite=True,
        )

        best_cfg = dict(smac.optimize())
        best_cfgs_by_fold.append(best_cfg)

        final_pipe = build_pipeline(best_cfg, seed=fold_seed, n_features=X_tr.shape[1])
        final_pipe.fit(X_tr, y_tr)

        fold_f1 = float(f1_score(y_va, final_pipe.predict(X_va)))
        outer_scores.append(fold_f1)

        print(
            f"[seed={seed}][{run_tag}] "
            f"outer fold {i}/{N_OUTER_SPLITS} -> F1={fold_f1:.6f}"
        )

    return {
        "f1_mean": float(np.mean(outer_scores)),
        "f1_std": float(np.std(outer_scores, ddof=1)) if len(outer_scores) > 1 else 0.0,
        "outer_fold_scores": [float(x) for x in outer_scores],
        "best_cfgs_by_outer_fold": best_cfgs_by_fold,
    }


def fit_final_rf_model(X, y, hp_name, seed, seed_out_dir):
    inner_cv = StratifiedKFold(
        n_splits=N_INNER_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    final_smac_dir = os.path.join(seed_out_dir, "final_smac", hp_name, "rf")

    scenario = Scenario(
        rf_configspace(seed),
        n_trials=N_TRIALS,
        deterministic=True,
        seed=seed,
        output_directory=final_smac_dir,
    )

    smac = HyperparameterOptimizationFacade(
        scenario,
        smac_objective(X, y, inner_cv, seed),
        overwrite=True,
    )

    best_cfg = dict(smac.optimize())

    final_pipe = build_pipeline(best_cfg, seed=seed, n_features=X.shape[1])
    final_pipe.fit(X, y)

    selector = final_pipe.named_steps["select"]
    selected_cols = X.columns.to_numpy()[selector.get_support()].tolist()

    if len(selected_cols) == 0:
        selected_cols = list(X.columns)

    model_dir = os.path.join(seed_out_dir, "final_models", hp_name)
    os.makedirs(model_dir, exist_ok=True)

    pipeline_path = os.path.join(model_dir, "rf_pipeline.joblib")
    selector_path = os.path.join(model_dir, "rf_selector.joblib")
    clf_path = os.path.join(model_dir, "rf_classifier.joblib")

    joblib.dump(final_pipe, pipeline_path)
    joblib.dump(selector, selector_path)
    joblib.dump(final_pipe.named_steps["clf"], clf_path)

    return {
        "smac_dir": final_smac_dir,
        "pipeline_path": pipeline_path,
        "selector_path": selector_path,
        "classifier_path": clf_path,
        "selected_features": selected_cols,
        "best_cfg": best_cfg,
    }