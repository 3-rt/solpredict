import numpy as np
from sklearn.ensemble import RandomForestRegressor

from solpredict.training.cv import evaluate_predictions, run_kfold_cv


def test_evaluate_predictions_reports_fold_metrics() -> None:
    metrics = evaluate_predictions(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
    assert set(metrics) == {"r2", "rmse", "mae"}
    assert metrics["rmse"] > 0


def test_run_kfold_cv_returns_aggregate_statistics() -> None:
    x = np.random.default_rng(0).normal(size=(20, 4))
    y = np.random.default_rng(1).normal(size=20)
    metrics = run_kfold_cv(
        x,
        y,
        random_seed=42,
        cv_folds=5,
        estimator_factory=lambda: RandomForestRegressor(n_estimators=5, random_state=42),
    )
    assert metrics["n_folds"] == 5
    assert len(metrics["fold_scores"]) == 5
    assert "rmse_mean" in metrics
