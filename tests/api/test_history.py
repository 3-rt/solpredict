from solpredict.db.repositories import record_prediction


def test_history_returns_paginated_items_newest_first(client, db_session) -> None:
    record_prediction(
        db_session,
        smiles="CCO",
        rf_prediction=-0.1,
        nn_prediction=-0.2,
        rf_model_version_id=None,
        nn_model_version_id=None,
        descriptors={"molecular_weight": 46.07},
        molecule_name="Ethanol",
        client_ip=None,
    )

    response = client.get("/history?limit=50&offset=0")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["smiles"] == "CCO"


def test_history_supports_smiles_filter(client, db_session) -> None:
    record_prediction(
        db_session,
        smiles="CCO",
        rf_prediction=-0.1,
        nn_prediction=-0.2,
        rf_model_version_id=None,
        nn_model_version_id=None,
        descriptors={"molecular_weight": 46.07},
        molecule_name="Ethanol",
        client_ip=None,
    )
    record_prediction(
        db_session,
        smiles="O",
        rf_prediction=0.5,
        nn_prediction=0.4,
        rf_model_version_id=None,
        nn_model_version_id=None,
        descriptors={"molecular_weight": 18.0},
        molecule_name="Water",
        client_ip=None,
    )

    response = client.get("/history?smiles=O")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["smiles"] == "O"
