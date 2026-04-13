"""Prediction history API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.deps import get_db
from solpredict.db.repositories import get_recent_predictions

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]


@router.get("/history")
def history(
    db: DbSession,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    smiles: str | None = Query(default=None),
) -> dict[str, object]:
    items, total = get_recent_predictions(db, limit=limit, offset=offset, smiles=smiles)
    return {
        "items": [
            {
                "id": row.id,
                "smiles": row.smiles,
                "molecule_name": row.molecule_name,
                "rf_prediction": row.rf_prediction,
                "nn_prediction": row.nn_prediction,
                "descriptors": row.descriptors,
                "created_at": row.created_at.isoformat(),
                "rf_model_version": row.rf_model_version.version if row.rf_model_version else None,
                "nn_model_version": row.nn_model_version.version if row.nn_model_version else None,
            }
            for row in items
        ],
        "total": total,
    }
