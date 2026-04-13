from __future__ import annotations

from ast import literal_eval
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from solpredict.model import SolubilityMLP
from solpredict.tracking import configure_mlflow_tracking, start_nested_run, start_parent_run
from solpredict.training.cv import evaluate_predictions, run_kfold_cv

RF_DEFAULTS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}
NN_DEFAULTS: dict[str, Any] = {
    "hidden_dims": (512, 128),
    "dropout": 0.2,
    "lr": 1e-3,
    "batch_size": 64,
    "weight_decay": 1e-5,
    "epochs": 100,
}
NN_HIDDEN_DIM_CHOICES = {
    "512x128": (512, 128),
    "1024x256": (1024, 256),
    "512x256x64": (512, 256, 64),
}


@dataclass(slots=True)
class TuneResult:
    best_params: dict[str, Any]
    cv_metrics: dict[str, Any]
    source: str


def _rf_factory(params: dict[str, Any], random_seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(random_state=random_seed, n_jobs=-1, **params)


def _rf_estimator_factory(
    params: dict[str, Any],
    random_seed: int,
) -> Callable[[], RandomForestRegressor]:
    def factory() -> RandomForestRegressor:
        return _rf_factory(params, random_seed)

    return factory


def _fit_mlp_model(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.float32],
    *,
    params: dict[str, Any],
    random_seed: int,
) -> SolubilityMLP:
    torch.manual_seed(random_seed)
    model = SolubilityMLP(
        input_dim=x_train.shape[1],
        hidden_dims=tuple(params["hidden_dims"]),
        dropout=float(params["dropout"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    batch_size = int(params["batch_size"])

    model.train()
    for _ in range(int(params["epochs"])):
        indices = torch.randperm(len(x_train_t))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            optimizer.zero_grad()
            predictions = model(x_train_t[batch_idx])
            loss = loss_fn(predictions, y_train_t[batch_idx])
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def _train_mlp_once(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.float32],
    x_val: NDArray[np.float32],
    params: dict[str, Any],
    random_seed: int,
) -> NDArray[np.float64]:
    model = _fit_mlp_model(x_train, y_train, params=params, random_seed=random_seed)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(x_val_t).cpu().numpy()
    return np.asarray(predictions, dtype=float)


def _run_mlp_cv(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    *,
    params: dict[str, Any],
    cv_folds: int,
    random_seed: int,
    trial: optuna.Trial | None = None,
) -> dict[str, Any]:
    # Per-fold pruning: we report each fold's RMSE as an intermediate signal
    # so the pruner gets at least one data point per fold before the trial ends.
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    fold_scores: list[dict[str, float]] = []
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(x)):
        predictions = _train_mlp_once(
            x[train_idx],
            y[train_idx],
            x[val_idx],
            params,
            random_seed + fold_index,
        )
        fold_metrics = evaluate_predictions(
            np.asarray(y[val_idx], dtype=float),
            predictions,
        )
        fold_scores.append(fold_metrics)
        if trial is not None:
            trial.report(float(fold_metrics["rmse"]), step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned

    summary: dict[str, Any] = {"n_folds": cv_folds, "fold_scores": fold_scores}
    for metric in ("r2", "rmse", "mae"):
        values = np.asarray([row[metric] for row in fold_scores], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    return summary


def _coerce_best_params(model_name: str, best_params: dict[str, Any]) -> dict[str, Any]:
    if model_name != "nn":
        return best_params
    normalized = dict(best_params)
    hidden_dims = normalized.get("hidden_dims")
    if isinstance(hidden_dims, str):
        if hidden_dims in NN_HIDDEN_DIM_CHOICES:
            normalized["hidden_dims"] = NN_HIDDEN_DIM_CHOICES[hidden_dims]
        else:
            normalized["hidden_dims"] = tuple(literal_eval(hidden_dims))
    elif isinstance(hidden_dims, list):
        normalized["hidden_dims"] = tuple(hidden_dims)
    normalized.setdefault("epochs", int(NN_DEFAULTS["epochs"]))
    normalized.setdefault("batch_size", int(NN_DEFAULTS["batch_size"]))
    return normalized


def tune_models(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    *,
    models: tuple[str, ...],
    cv_folds: int,
    n_trials: int,
    random_seed: int,
    mlflow_tracking_uri: str,
    skip_tune: bool,
) -> dict[str, TuneResult]:
    configure_mlflow_tracking(mlflow_tracking_uri)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results: dict[str, TuneResult] = {}

    for model_name in models:
        defaults = dict(RF_DEFAULTS if model_name == "rf" else NN_DEFAULTS)
        if skip_tune:
            if model_name == "rf":
                metrics = run_kfold_cv(
                    np.asarray(x, dtype=float),
                    np.asarray(y, dtype=float),
                    random_seed=random_seed,
                    cv_folds=cv_folds,
                    estimator_factory=_rf_estimator_factory(defaults, random_seed),
                )
            else:
                metrics = _run_mlp_cv(
                    x,
                    y,
                    params=defaults,
                    cv_folds=cv_folds,
                    random_seed=random_seed,
                )
            results[model_name] = TuneResult(
                best_params=defaults,
                cv_metrics=metrics,
                source="defaults",
            )
            continue

        with start_parent_run(f"tune_{model_name}"):
            study_kwargs: dict[str, Any] = {
                "direction": "minimize",
                "sampler": optuna.samplers.TPESampler(seed=random_seed),
            }
            if model_name == "nn":
                # MLP reports one intermediate value per CV fold (5 folds total), so
                # warmup must be <5 for pruning to ever activate. Prune starting after
                # the second fold once at least 5 trials have established a baseline.
                study_kwargs["pruner"] = optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=2,
                )
            study = optuna.create_study(**study_kwargs)

            def objective(trial: optuna.Trial, model_name: str = model_name) -> float:
                with start_nested_run(f"{model_name}_trial_{trial.number}"):
                    if model_name == "rf":
                        params: dict[str, Any] = {
                            "n_estimators": trial.suggest_categorical(
                                "n_estimators",
                                [100, 200, 500],
                            ),
                            "max_depth": trial.suggest_categorical(
                                "max_depth",
                                [None, 5, 10, 20, 30],
                            ),
                            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                            "max_features": trial.suggest_categorical(
                                "max_features",
                                ["sqrt", "log2", 0.5],
                            ),
                        }
                        metrics = run_kfold_cv(
                            np.asarray(x, dtype=float),
                            np.asarray(y, dtype=float),
                            random_seed=random_seed,
                            cv_folds=cv_folds,
                            estimator_factory=_rf_estimator_factory(params, random_seed),
                        )
                    else:
                        params = {
                            "hidden_dims": trial.suggest_categorical(
                                "hidden_dims",
                                list(NN_HIDDEN_DIM_CHOICES),
                            ),
                            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                            "weight_decay": trial.suggest_float(
                                "weight_decay",
                                1e-6,
                                1e-3,
                                log=True,
                            ),
                            "epochs": int(NN_DEFAULTS["epochs"]),
                        }
                        params["hidden_dims"] = NN_HIDDEN_DIM_CHOICES[str(params["hidden_dims"])]
                        metrics = _run_mlp_cv(
                            x,
                            y,
                            params=params,
                            cv_folds=cv_folds,
                            random_seed=random_seed,
                            trial=trial,
                        )

                    mlflow.log_params({key: str(value) for key, value in params.items()})
                    mlflow.log_metric("rmse_mean", float(metrics["rmse_mean"]))
                    trial.set_user_attr("cv_metrics", metrics)
                    return float(metrics["rmse_mean"])

            study.optimize(objective, n_trials=n_trials)
            best_params = _coerce_best_params(model_name, dict(study.best_params))
            best_metrics = dict(study.best_trial.user_attrs["cv_metrics"])
            results[model_name] = TuneResult(
                best_params=best_params,
                cv_metrics=best_metrics,
                source="optuna",
            )

    return results
