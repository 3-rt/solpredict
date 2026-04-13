from pathlib import Path

import numpy as np

from solpredict.training.tune import tune_models


def test_tune_models_returns_best_params_for_selected_models(tmp_path: Path) -> None:
    x = np.random.default_rng(0).normal(size=(24, 8)).astype("float32")
    y = np.random.default_rng(1).normal(size=24).astype("float32")
    results = tune_models(
        x,
        y,
        models=("rf", "nn"),
        cv_folds=3,
        n_trials=2,
        random_seed=42,
        mlflow_tracking_uri=f"file://{tmp_path / 'mlruns'}",
        skip_tune=False,
    )
    assert set(results) == {"rf", "nn"}
    assert results["rf"].best_params
    assert results["nn"].best_params


def test_tune_models_skip_tune_uses_known_good_defaults(tmp_path: Path) -> None:
    x = np.random.default_rng(2).normal(size=(24, 8)).astype("float32")
    y = np.random.default_rng(3).normal(size=24).astype("float32")
    results = tune_models(
        x,
        y,
        models=("rf",),
        cv_folds=3,
        n_trials=2,
        random_seed=42,
        mlflow_tracking_uri=f"file://{tmp_path / 'mlruns'}",
        skip_tune=True,
    )
    assert results["rf"].source == "defaults"
