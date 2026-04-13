from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from .models import ModelVersion, Prediction


def upsert_model_version(
    session: Session,
    *,
    name: str,
    version: str,
    artifact_path: str,
    mlflow_run_id: str | None,
    trained_at: datetime | None,
    cv_r2_mean: float | None,
    cv_rmse_mean: float | None,
    test_r2: float | None,
    test_rmse: float | None,
    hyperparameters: dict[str, Any],
) -> ModelVersion:
    existing = session.scalar(
        select(ModelVersion).where(ModelVersion.name == name, ModelVersion.version == version)
    )
    if existing is None:
        existing = ModelVersion(
            name=name,
            version=version,
            artifact_path=artifact_path,
        )
        session.add(existing)
        session.flush()

    existing.artifact_path = artifact_path
    existing.mlflow_run_id = mlflow_run_id
    existing.trained_at = trained_at or datetime.now(UTC)
    existing.cv_r2_mean = cv_r2_mean
    existing.cv_rmse_mean = cv_rmse_mean
    existing.test_r2 = test_r2
    existing.test_rmse = test_rmse
    existing.hyperparameters = hyperparameters
    existing.is_active = True

    session.execute(
        update(ModelVersion)
        .where(ModelVersion.name == name, ModelVersion.id != existing.id)
        .values(is_active=False)
    )
    session.commit()
    session.refresh(existing)
    return existing


def get_active_model(session: Session, name: str) -> ModelVersion | None:
    return session.scalar(
        select(ModelVersion).where(ModelVersion.name == name, ModelVersion.is_active.is_(True))
    )


def record_prediction(
    session: Session,
    *,
    smiles: str,
    rf_prediction: float | None,
    nn_prediction: float | None,
    rf_model_version_id: int | None,
    nn_model_version_id: int | None,
    descriptors: dict[str, Any],
    molecule_name: str | None,
    client_ip: str | None,
) -> Prediction:
    row = Prediction(
        smiles=smiles,
        rf_prediction=rf_prediction,
        nn_prediction=nn_prediction,
        rf_model_version_id=rf_model_version_id,
        nn_model_version_id=nn_model_version_id,
        descriptors=descriptors,
        molecule_name=molecule_name,
        client_ip=client_ip,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def get_recent_predictions(
    session: Session,
    *,
    limit: int = 50,
    offset: int = 0,
    smiles: str | None = None,
) -> tuple[list[Prediction], int]:
    stmt = select(Prediction)
    count_stmt = select(func.count()).select_from(Prediction)
    if smiles:
        stmt = stmt.where(Prediction.smiles == smiles)
        count_stmt = count_stmt.where(Prediction.smiles == smiles)
    stmt = (
        stmt.order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(session.scalars(stmt)), int(session.scalar(count_stmt) or 0)


def list_model_versions(session: Session, *, limit: int = 10) -> list[ModelVersion]:
    stmt = (
        select(ModelVersion)
        .order_by(
            ModelVersion.is_active.desc(),
            ModelVersion.trained_at.desc(),
            ModelVersion.id.desc(),
        )
        .limit(limit)
    )
    return list(session.scalars(stmt))
