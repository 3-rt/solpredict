"""Prediction-related API routes."""

from __future__ import annotations

import logging
from typing import Annotated

import numpy as np
import torch
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.deps import get_db
from solpredict.db.repositories import record_prediction
from solpredict.featurize import smiles_to_descriptors, smiles_to_fingerprint

logger = logging.getLogger(__name__)
router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]

KNOWN_MOLECULES = {
    "CCO": "Ethanol",
    "CC(=O)Oc1ccccc1C(=O)O": "Aspirin",
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O": "Caffeine",
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O": "Glucose",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O": "Ibuprofen",
    "c1ccccc1": "Benzene",
    "CC(=O)O": "Acetic acid",
    "O": "Water",
    "CCCCCCCCCCCC": "Dodecane",
    "c1ccc2cc3ccccc3cc2c1": "Anthracene",
}


class PredictRequest(BaseModel):
    smiles: str


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    return {
        "status": "ok",
        "models_loaded": {
            "random_forest": request.app.state.rf_model is not None,
            "neural_network": request.app.state.nn_model is not None,
        },
    }


@router.post("/predict")
def predict(
    req: PredictRequest,
    request: Request,
    db: DbSession,
) -> dict[str, object]:
    smiles = req.smiles.strip()

    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return {"smiles": smiles, "valid": False, "error": "Could not parse SMILES string"}

    descriptors = smiles_to_descriptors(smiles)

    predictions: dict[str, float] = {}
    rf_model = request.app.state.rf_model
    nn_model = request.app.state.nn_model

    if rf_model is not None:
        rf_pred = rf_model.predict(fp.reshape(1, -1).astype(np.float32))[0]
        predictions["random_forest"] = round(float(rf_pred), 4)
    if nn_model is not None:
        with torch.no_grad():
            x = torch.FloatTensor(fp.astype(np.float32)).unsqueeze(0)
            nn_pred = nn_model(x).item()
            predictions["neural_network"] = round(float(nn_pred), 4)

    try:
        record_prediction(
            db,
            smiles=smiles,
            rf_prediction=predictions.get("random_forest"),
            nn_prediction=predictions.get("neural_network"),
            rf_model_version_id=getattr(request.app.state.rf_model_version, "id", None),
            nn_model_version_id=getattr(request.app.state.nn_model_version, "id", None),
            descriptors=descriptors or {},
            molecule_name=KNOWN_MOLECULES.get(smiles),
            client_ip=request.client.host if request.client else None,
        )
    except Exception:
        logger.warning("Failed to record prediction", exc_info=True, extra={"smiles": smiles})

    return {
        "smiles": smiles,
        "valid": True,
        "predictions": predictions,
        "descriptors": descriptors,
        "molecule_name": KNOWN_MOLECULES.get(smiles),
    }


@router.get("/examples")
def examples(request: Request) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    rf_model = request.app.state.rf_model
    nn_model = request.app.state.nn_model

    for smiles, name in KNOWN_MOLECULES.items():
        fp = smiles_to_fingerprint(smiles)
        if fp is None:
            continue
        descriptors = smiles_to_descriptors(smiles)
        predictions: dict[str, float] = {}
        if rf_model is not None:
            predictions["random_forest"] = round(
                float(rf_model.predict(fp.reshape(1, -1).astype(np.float32))[0]),
                4,
            )
        if nn_model is not None:
            with torch.no_grad():
                x = torch.FloatTensor(fp.astype(np.float32)).unsqueeze(0)
                predictions["neural_network"] = round(float(nn_model(x).item()), 4)
        results.append(
            {
                "smiles": smiles,
                "name": name,
                "predictions": predictions,
                "descriptors": descriptors,
            }
        )

    return results
