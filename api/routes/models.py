"""Model version API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.deps import get_db
from solpredict.db.repositories import list_model_versions

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]


@router.get("/models")
def models(db: DbSession) -> list[dict[str, object]]:
    return [
        {
            "id": row.id,
            "name": row.name,
            "version": row.version,
            "mlflow_run_id": row.mlflow_run_id,
            "artifact_path": row.artifact_path,
            "trained_at": row.trained_at.isoformat() if row.trained_at else None,
            "cv_r2_mean": row.cv_r2_mean,
            "cv_rmse_mean": row.cv_rmse_mean,
            "test_r2": row.test_r2,
            "test_rmse": row.test_rmse,
            "hyperparameters": row.hyperparameters,
            "is_active": row.is_active,
        }
        for row in list_model_versions(db, limit=10)
    ]
