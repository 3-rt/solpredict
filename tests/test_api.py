import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    """Health endpoint should return OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_smiles(client):
    """Valid SMILES should return predictions from both models."""
    response = client.post("/predict", json={"smiles": "CCO"})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert "random_forest" in data["predictions"]
    assert "neural_network" in data["predictions"]
    assert isinstance(data["predictions"]["random_forest"], float)
    assert isinstance(data["predictions"]["neural_network"], float)
    assert "descriptors" in data
    assert "molecular_weight" in data["descriptors"]


def test_predict_invalid_smiles(client):
    """Invalid SMILES should return valid=false with error."""
    response = client.post("/predict", json={"smiles": "NOT_VALID"})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert "error" in data


def test_examples(client):
    """Examples endpoint should return a list of pre-computed predictions."""
    response = client.get("/examples")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 5
    assert "smiles" in data[0]
    assert "predictions" in data[0]
    assert "name" in data[0]
