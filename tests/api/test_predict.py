from api.main import create_app
from solpredict.db.models import Prediction


def test_app_factory_boots_with_db_override(client) -> None:
    app = create_app()
    assert app.title == "SolPredict API"


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_smiles_records_history(client, db_session) -> None:
    response = client.post("/predict", json={"smiles": "CCO"})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert "random_forest" in data["predictions"]
    assert "neural_network" in data["predictions"]
    rows = db_session.query(Prediction).all()
    assert len(rows) == 1
    assert rows[0].smiles == "CCO"
    assert rows[0].molecule_name == "Ethanol"


def test_predict_invalid_smiles_does_not_persist(client, db_session) -> None:
    response = client.post("/predict", json={"smiles": "NOT_VALID"})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert "error" in data
    assert db_session.query(Prediction).count() == 0


def test_predict_survives_repository_write_failure(client, monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr("api.routes.predict.record_prediction", _raise)
    response = client.post("/predict", json={"smiles": "CCO"})
    assert response.status_code == 200
    assert response.json()["valid"] is True


def test_examples(client) -> None:
    response = client.get("/examples")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 5
    assert "smiles" in data[0]
    assert "predictions" in data[0]
    assert "name" in data[0]
