import os
from functools import partial

import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_regression

from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
)

from src.meta_learning.regression.config import (
    N_TRIALS,
    N_OUTER_SPLITS,
    N_INNER_SPLITS,
    K_MIN,
    K_MAX,
    OPTIMIZE,
)


def get_rf_configspace(seed: int) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)
    cs.add_hyperparameters([
        UniformIntegerHyperparameter("n_estimators", 100, 1500, default_value=600),
        UniformIntegerHyperparameter("max_depth", 2, 50, default_value=20),
        UniformIntegerHyperparameter("min_samples_split", 2, 30, default_value=2),
        UniformIntegerHyperparameter("min_samples_leaf", 1, 30, default_value=1),
        CategoricalHyperparameter("max_features", ["sqrt", "log2", None]),
        CategoricalHyperparameter("bootstrap", [True, False]),
        CategoricalHyperparameter("criterion", ["squared_error", "absolute_error"]),
        UniformIntegerHyperparameter("k_best", K_MIN, K_MAX),
    ])
    return cs


def make_rf(cfg: dict, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(cfg["n_estimators"]),
        max_depth=int(cfg["max_depth"]),
        min_samples_split=int(cfg["min_samples_split"]),
        min_samples_leaf=int(cfg["min_samples_leaf"]),
        max_features=cfg["max_features"],
        bootstrap=bool(cfg["bootstrap"]),
        criterion=str(cfg["criterion"]),
        random_state=seed,
        n_jobs=-1,
    )


def build_pipeline(cfg: dict, seed: int, n_features: int) -> Pipeline:
    k = min(int(cfg["k_best"]), n_features)
    mi_func = partial(mutual_info_regression, random_state=seed)

    return Pipeline([
        ("select", SelectKBest(score_func=mi_func, k=k)),
        ("rf", make_rf(cfg, seed=seed)),
    ])


def scoring_dict():
    return {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
    }


def compute_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }


def nested_cv_for_one_hpi(X_df, y, hp_name, seed, seed_out_dir):
    scoring = scoring_dict()
    key = OPTIMIZE.lower()

    outer_cv = KFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    outer_fold_metrics = []
    outer_fold_best_cfgs = []

    for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X_df), start=1):
        X_tr, X_te = X_df.iloc[tr_idx], X_df.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        fold_seed = seed + fold_i

        inner_cv = KFold(
            n_splits=N_INNER_SPLITS,
            shuffle=True,
            random_state=fold_seed,
        )

        def objective(cfg, seed: int = 0, **kwargs) -> float:
            cfg = dict(cfg)
            pipe = build_pipeline(cfg, seed=fold_seed, n_features=X_tr.shape[1])

            scores = cross_validate(
                pipe,
                X_tr,
                y_tr.to_numpy(dtype=float),
                cv=inner_cv,
                scoring={key: scoring[key]},
                n_jobs=-1,
                error_score="raise",
                return_train_score=False,
            )

            return float(-np.mean(scores[f"test_{key}"]))

        scenario = Scenario(
            get_rf_configspace(seed=fold_seed),
            n_trials=N_TRIALS,
            deterministic=True,
            seed=fold_seed,
            output_directory=os.path.join(
                seed_out_dir,
                f"nested_smac_{hp_name}",
                f"outer_fold_{fold_i}",
            ),
        )

        smac = HyperparameterOptimizationFacade(
            scenario,
            objective,
            overwrite=True,
        )

        best_cfg = dict(smac.optimize())
        outer_fold_best_cfgs.append(best_cfg)

        final_pipe = build_pipeline(best_cfg, seed=fold_seed, n_features=X_tr.shape[1])
        final_pipe.fit(X_tr, y_tr.to_numpy(dtype=float))

        y_pred = final_pipe.predict(X_te)
        y_true = y_te.to_numpy(dtype=float)

        metrics = compute_metrics(y_true, y_pred)
        outer_fold_metrics.append(metrics)

        print(
            f"[seed={seed}][{hp_name}] Outer fold {fold_i}/{N_OUTER_SPLITS} -> "
            f"MAE={metrics['mae']:.6f} "
            f"MSE={metrics['mse']:.6f} "
            f"RMSE={metrics['rmse']:.6f}"
        )

    mae_values = [m["mae"] for m in outer_fold_metrics]
    mse_values = [m["mse"] for m in outer_fold_metrics]
    rmse_values = [m["rmse"] for m in outer_fold_metrics]

    return {
        "seed": seed,
        "outer_folds": outer_fold_metrics,
        "nested_mae_mean": float(np.mean(mae_values)),
        "nested_mae_std": float(np.std(mae_values, ddof=1)) if len(mae_values) > 1 else 0.0,
        "nested_mse_mean": float(np.mean(mse_values)),
        "nested_rmse_mean": float(np.mean(rmse_values)),
        "best_cfgs_by_outer_fold": outer_fold_best_cfgs,
        "optimized_metric": key,
    }


def fit_final_model(X_df, y, hp_name, seed, seed_out_dir):
    key = OPTIMIZE.lower()
    scoring = scoring_dict()

    inner_cv = KFold(
        n_splits=N_INNER_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    def objective(cfg, seed: int = 0, **kwargs) -> float:
        cfg = dict(cfg)
        pipe = build_pipeline(cfg, seed=seed, n_features=X_df.shape[1])

        scores = cross_validate(
            pipe,
            X_df,
            y.to_numpy(dtype=float),
            cv=inner_cv,
            scoring={key: scoring[key]},
            n_jobs=-1,
            error_score="raise",
            return_train_score=False,
        )

        return float(-np.mean(scores[f"test_{key}"]))

    scenario = Scenario(
        get_rf_configspace(seed=seed),
        n_trials=N_TRIALS,
        deterministic=True,
        seed=seed,
        output_directory=os.path.join(seed_out_dir, f"final_smac_{hp_name}"),
    )

    smac = HyperparameterOptimizationFacade(
        scenario,
        objective,
        overwrite=True,
    )

    best_cfg = dict(smac.optimize())

    final_pipe = build_pipeline(best_cfg, seed=seed, n_features=X_df.shape[1])
    final_pipe.fit(X_df, y.to_numpy(dtype=float))

    selector = final_pipe.named_steps["select"]
    selected_cols = X_df.columns.to_numpy()[selector.get_support()].tolist()

    if len(selected_cols) == 0:
        selected_cols = list(X_df.columns)

    model_path = os.path.join(seed_out_dir, f"rf_model__{hp_name}.joblib")
    selector_path = os.path.join(seed_out_dir, f"selector__{hp_name}.joblib")
    pipeline_path = os.path.join(seed_out_dir, f"pipeline__{hp_name}.joblib")

    joblib.dump(final_pipe.named_steps["rf"], model_path)
    joblib.dump(final_pipe.named_steps["select"], selector_path)
    joblib.dump(final_pipe, pipeline_path)

    return {
        "seed": seed,
        "model_path": model_path,
        "selector_path": selector_path,
        "pipeline_path": pipeline_path,
        "selected_features": selected_cols,
        "smac_incumbent": best_cfg,
    }