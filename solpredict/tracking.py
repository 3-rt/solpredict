from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np
import torch
from mlflow.models import infer_signature

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sklearn.base import BaseEstimator


def configure_mlflow_tracking(tracking_uri: str) -> str:
    """Point MLflow at the configured backend, creating local file stores when needed."""
    if tracking_uri.startswith("file://"):
        Path(tracking_uri.removeprefix("file://")).mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The filesystem tracking backend .* is deprecated.*",
            category=FutureWarning,
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Default")
    return mlflow.get_tracking_uri()


def start_parent_run(run_name: str) -> Any:
    """Start a top-level MLflow run."""
    return mlflow.start_run(run_name=run_name)


def start_nested_run(run_name: str) -> Any:
    """Start a nested MLflow child run."""
    return mlflow.start_run(run_name=run_name, nested=True)


def log_summary_metrics(metrics: dict[str, float]) -> None:
    """Write a batch of scalar summary metrics to the active MLflow run."""
    mlflow.log_metrics(metrics)


def log_sklearn_model(
    model: BaseEstimator,
    x_sample: NDArray[Any],
    *,
    artifact_path: str = "model",
) -> str:
    """Log a scikit-learn model to the active MLflow run and return its run_id."""
    x_example = np.asarray(x_sample[:5], dtype=float)
    predictions = model.predict(x_example)
    signature = infer_signature(x_example, predictions)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        signature=signature,
        input_example=x_example,
    )
    active = mlflow.active_run()
    assert active is not None, "log_sklearn_model must be called inside an active MLflow run"
    return str(active.info.run_id)


def log_pytorch_model(
    model: torch.nn.Module,
    x_sample: NDArray[Any],
    *,
    artifact_path: str = "model",
) -> str:
    """Log a PyTorch model to the active MLflow run and return its run_id."""
    x_example = np.asarray(x_sample[:5], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(x_example, dtype=torch.float32)).cpu().numpy()
    signature = infer_signature(x_example, predictions)
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=artifact_path,
        signature=signature,
        input_example=x_example,
    )
    active = mlflow.active_run()
    assert active is not None, "log_pytorch_model must be called inside an active MLflow run"
    return str(active.info.run_id)
