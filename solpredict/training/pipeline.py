from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy.orm import Session

from solpredict.config import Settings, get_settings
from solpredict.db.repositories import upsert_model_version
from solpredict.model import SolubilityMLP
from solpredict.tracking import (
    configure_mlflow_tracking,
    log_pytorch_model,
    log_sklearn_model,
    start_parent_run,
)
from solpredict.training.cv import evaluate_predictions
from solpredict.training.data import featurize_dataset, load_esol, split_holdout
from solpredict.training.tune import NN_DEFAULTS, RF_DEFAULTS, tune_models


@dataclass(slots=True)
class PipelineOutcome:
    rf_model_path: Path
    nn_model_path: Path
    results_path: Path
    model_versions: dict[str, str]


def _train_rf(
    x_train: NDArray[np.int8],
    y_train: NDArray[np.float64],
    *,
    params: dict[str, Any],
    random_seed: int,
) -> RandomForestRegressor:
    model = RandomForestRegressor(random_state=random_seed, n_jobs=-1, **params)
    model.fit(np.asarray(x_train, dtype=float), y_train)
    return model


def _train_nn(
    x_train: NDArray[np.int8],
    y_train: NDArray[np.float64],
    *,
    params: dict[str, Any],
    random_seed: int,
) -> tuple[SolubilityMLP, list[float]]:
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
    x_train_t = torch.tensor(np.asarray(x_train, dtype=np.float32), dtype=torch.float32)
    y_train_t = torch.tensor(np.asarray(y_train, dtype=np.float32), dtype=torch.float32)
    batch_size = int(params["batch_size"])
    training_losses: list[float] = []

    model.train()
    for _ in range(int(params["epochs"])):
        indices = torch.randperm(len(x_train_t))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            optimizer.zero_grad()
            predictions = model(x_train_t[batch_idx])
            loss = loss_fn(predictions, y_train_t[batch_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        training_losses.append(round(epoch_loss / max(n_batches, 1), 4))

    model.eval()
    return model, training_losses


def _predict_nn(model: SolubilityMLP, x: NDArray[np.int8]) -> NDArray[np.float64]:
    x_tensor = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
    with torch.no_grad():
        predictions = model(x_tensor).cpu().numpy()
    return np.asarray(predictions, dtype=float)


def _model_version_label(model_key: str, now: datetime) -> str:
    suffix = "rf" if model_key == "rf" else "nn"
    return now.strftime(f"%Y.%m.%d-%H%M%S-{suffix}")


def _nn_architecture(input_dim: int, hidden_dims: tuple[int, ...]) -> str:
    return " \u2192 ".join(str(dim) for dim in (input_dim, *hidden_dims, 1))


def _build_results_payload(
    *,
    settings: Settings,
    frame_size: int,
    y_test: NDArray[np.float64],
    rf_pred: NDArray[np.float64],
    nn_pred: NDArray[np.float64],
    rf_section: dict[str, Any] | None,
    nn_section: dict[str, Any] | None,
    rf_model: RandomForestRegressor | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset": {
            "name": "ESOL (Delaney)",
            "n_molecules": frame_size,
            "target": "log(solubility) in mol/L",
            "split": (
                f"{int((1 - settings.train_test_size) * 100)}/"
                f"{int(settings.train_test_size * 100)} train/test"
            ),
            "random_seed": settings.random_seed,
        },
        "models": {},
        "plots": {
            "scatter": {
                "y_true": y_test.tolist(),
                "rf_pred": rf_pred.tolist(),
                "nn_pred": nn_pred.tolist(),
            },
            "residuals": {
                "rf_residuals": (y_test - rf_pred).tolist(),
                "nn_residuals": (y_test - nn_pred).tolist(),
            },
        },
    }
    if rf_section is not None:
        payload["models"]["random_forest"] = rf_section
    if nn_section is not None:
        payload["models"]["neural_network"] = nn_section
    if rf_model is not None and hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
        top_indices = np.argsort(importances)[-20:][::-1]
        payload["feature_importance"] = {
            "bit_positions": top_indices.tolist(),
            "importances": importances[top_indices].tolist(),
        }
    return payload


def run_training_pipeline(
    *,
    db_session: Session,
    esol_csv_path: str | Path | None = None,
    models: tuple[str, ...] = ("rf", "nn"),
    skip_tune: bool = False,
    n_trials: int | None = None,
) -> PipelineOutcome:
    settings = get_settings()
    configure_mlflow_tracking(settings.mlflow_tracking_uri)
    frame = load_esol(esol_csv_path or settings.esol_csv_path)
    featurized = featurize_dataset(
        frame,
        cache_dir=settings.cache_dir,
        fp_radius=settings.fp_radius,
        fp_nbits=settings.fp_nbits,
    )
    targets = frame.loc[featurized.valid_mask, "log_solubility"].to_numpy(dtype=float)
    x_train, x_test, y_train, y_test = split_holdout(
        featurized.fingerprints,
        targets,
        test_size=settings.train_test_size,
        random_seed=settings.random_seed,
    )
    tuned = tune_models(
        np.asarray(x_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
        models=models,
        cv_folds=settings.cv_folds,
        n_trials=n_trials or settings.optuna_trials,
        random_seed=settings.random_seed,
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
        skip_tune=skip_tune,
    )

    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(settings.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    rf_model_path = model_dir / "random_forest.pkl"
    nn_model_path = model_dir / "neural_network.pt"

    version_labels: dict[str, str] = {}
    now = datetime.now(UTC)
    rf_pred = np.zeros_like(y_test)
    nn_pred = np.zeros_like(y_test)
    rf_section: dict[str, Any] | None = None
    nn_section: dict[str, Any] | None = None
    rf_model: RandomForestRegressor | None = None

    if "rf" in models:
        rf_params = dict(tuned["rf"].best_params if not skip_tune else RF_DEFAULTS)
        rf_model = _train_rf(
            x_train,
            y_train,
            params=rf_params,
            random_seed=settings.random_seed,
        )
        rf_train_pred = np.asarray(rf_model.predict(np.asarray(x_train, dtype=float)), dtype=float)
        rf_pred = np.asarray(rf_model.predict(np.asarray(x_test, dtype=float)), dtype=float)
        rf_train_metrics = evaluate_predictions(y_train, rf_train_pred)
        rf_test_metrics = evaluate_predictions(y_test, rf_pred)
        joblib.dump(rf_model, rf_model_path)
        rf_version = _model_version_label("rf", now)
        with start_parent_run(f"retrain_{rf_version}"):
            rf_run_id = log_sklearn_model(
                rf_model,
                np.asarray(x_train, dtype=float),
            )
            upsert_model_version(
                db_session,
                name="random_forest",
                version=rf_version,
                artifact_path=str(rf_model_path),
                mlflow_run_id=rf_run_id,
                trained_at=now,
                cv_r2_mean=tuned["rf"].cv_metrics.get("r2_mean"),
                cv_rmse_mean=tuned["rf"].cv_metrics.get("rmse_mean"),
                test_r2=rf_test_metrics["r2"],
                test_rmse=rf_test_metrics["rmse"],
                hyperparameters=rf_params,
            )
        version_labels["rf"] = rf_version
        rf_section = {
            "name": "Random Forest",
            "library": "scikit-learn",
            "params": rf_params,
            "train_metrics": rf_train_metrics,
            "test_metrics": rf_test_metrics,
            "cv_metrics": tuned["rf"].cv_metrics,
        }

    if "nn" in models:
        nn_params = dict(tuned["nn"].best_params if not skip_tune else NN_DEFAULTS)
        nn_model, nn_losses = _train_nn(
            x_train,
            y_train,
            params=nn_params,
            random_seed=settings.random_seed,
        )
        nn_train_pred = _predict_nn(nn_model, x_train)
        nn_pred = _predict_nn(nn_model, x_test)
        nn_train_metrics = evaluate_predictions(y_train, nn_train_pred)
        nn_test_metrics = evaluate_predictions(y_test, nn_pred)
        torch.save(nn_model.state_dict(), nn_model_path)
        nn_version = _model_version_label("nn", now)
        with start_parent_run(f"retrain_{nn_version}"):
            nn_run_id = log_pytorch_model(
                nn_model,
                np.asarray(x_train, dtype=np.float32),
            )
            upsert_model_version(
                db_session,
                name="neural_network",
                version=nn_version,
                artifact_path=str(nn_model_path),
                mlflow_run_id=nn_run_id,
                trained_at=now,
                cv_r2_mean=tuned["nn"].cv_metrics.get("r2_mean"),
                cv_rmse_mean=tuned["nn"].cv_metrics.get("rmse_mean"),
                test_r2=nn_test_metrics["r2"],
                test_rmse=nn_test_metrics["rmse"],
                hyperparameters=nn_params,
            )
        version_labels["nn"] = nn_version
        nn_section = {
            "name": "Neural Network",
            "library": "PyTorch",
            "architecture": _nn_architecture(settings.fp_nbits, tuple(nn_params["hidden_dims"])),
            "params": nn_params,
            "train_metrics": nn_train_metrics,
            "test_metrics": nn_test_metrics,
            "cv_metrics": tuned["nn"].cv_metrics,
            "training_losses": nn_losses,
        }

    results_payload = _build_results_payload(
        settings=settings,
        frame_size=len(frame),
        y_test=y_test,
        rf_pred=rf_pred,
        nn_pred=nn_pred,
        rf_section=rf_section,
        nn_section=nn_section,
        rf_model=rf_model,
    )
    results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    return PipelineOutcome(
        rf_model_path=rf_model_path,
        nn_model_path=nn_model_path,
        results_path=results_path,
        model_versions=version_labels,
    )
