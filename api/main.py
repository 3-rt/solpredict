# api/main.py
"""FastAPI prediction service for SolPredict."""

import os
import sys
from contextlib import asynccontextmanager

import joblib
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solpredict.featurize import smiles_to_descriptors, smiles_to_fingerprint
from solpredict.model import SolubilityMLP

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

rf_model = None
nn_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global rf_model, nn_model

    rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    nn_path = os.path.join(MODEL_DIR, "neural_network.pt")

    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
    if os.path.exists(nn_path):
        nn_model = SolubilityMLP(input_dim=2048)
        nn_model.load_state_dict(torch.load(nn_path, map_location="cpu", weights_only=True))
        nn_model.eval()

    yield


app = FastAPI(title="SolPredict API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": {
            "random_forest": rf_model is not None,
            "neural_network": nn_model is not None,
        },
    }


@app.post("/predict")
def predict(req: PredictRequest):
    smiles = req.smiles.strip()

    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return {"smiles": smiles, "valid": False, "error": "Could not parse SMILES string"}

    descriptors = smiles_to_descriptors(smiles)

    predictions = {}
    if rf_model is not None:
        rf_pred = rf_model.predict(fp.reshape(1, -1).astype(np.float32))[0]
        predictions["random_forest"] = round(float(rf_pred), 4)
    if nn_model is not None:
        with torch.no_grad():
            x = torch.FloatTensor(fp.astype(np.float32)).unsqueeze(0)
            nn_pred = nn_model(x).item()
            predictions["neural_network"] = round(float(nn_pred), 4)

    return {
        "smiles": smiles,
        "valid": True,
        "predictions": predictions,
        "descriptors": descriptors,
        "molecule_name": KNOWN_MOLECULES.get(smiles),
    }


@app.get("/examples")
def examples():
    """Return pre-computed predictions for well-known molecules."""
    results = []
    for smiles, name in KNOWN_MOLECULES.items():
        fp = smiles_to_fingerprint(smiles)
        if fp is None:
            continue
        descriptors = smiles_to_descriptors(smiles)
        predictions = {}
        if rf_model is not None:
            predictions["random_forest"] = round(
                float(rf_model.predict(fp.reshape(1, -1).astype(np.float32))[0]), 4
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
