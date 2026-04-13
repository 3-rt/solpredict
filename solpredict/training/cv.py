from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


def evaluate_predictions(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> dict[str, float]:
    """Compute the regression metrics shared across CV and holdout evaluation."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def run_kfold_cv(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    random_seed: int,
    cv_folds: int,
    estimator_factory: Callable[[], Any],
) -> dict[str, Any]:
    """Fit a fresh estimator per fold and return per-fold plus aggregate metrics."""
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    fold_scores: list[dict[str, float]] = []

    for train_idx, val_idx in splitter.split(x):
        estimator = estimator_factory()
        estimator.fit(x[train_idx], y[train_idx])
        predictions = np.asarray(estimator.predict(x[val_idx]), dtype=float)
        fold_scores.append(evaluate_predictions(y[val_idx], predictions))

    summary: dict[str, Any] = {
        "n_folds": cv_folds,
        "fold_scores": fold_scores,
    }
    for metric in ("r2", "rmse", "mae"):
        values = np.asarray([row[metric] for row in fold_scores], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    return summary
