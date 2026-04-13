from solpredict.db.models import ModelVersion
from solpredict.db.repositories import (
    get_active_model,
    get_recent_predictions,
    list_model_versions,
    record_prediction,
    upsert_model_version,
)


def test_upsert_model_version_demotes_previous_active(db_session) -> None:
    first = upsert_model_version(
        db_session,
        name="random_forest",
        version="2026.04.12-rf",
        artifact_path="models/random_forest.pkl",
        mlflow_run_id=None,
        trained_at=None,
        cv_r2_mean=0.71,
        cv_rmse_mean=1.17,
        test_r2=0.70,
        test_rmse=1.20,
        hyperparameters={"n_estimators": 100},
    )
    second = upsert_model_version(
        db_session,
        name="random_forest",
        version="2026.04.13-rf",
        artifact_path="models/random_forest_v2.pkl",
        mlflow_run_id=None,
        trained_at=None,
        cv_r2_mean=0.73,
        cv_rmse_mean=1.10,
        test_r2=0.72,
        test_rmse=1.12,
        hyperparameters={"n_estimators": 200},
    )

    db_session.refresh(first)
    assert first.is_active is False
    active = get_active_model(db_session, "random_forest")
    assert active is not None
    assert active.id == second.id


def test_record_prediction_round_trips_descriptor_payload(db_session) -> None:
    row = record_prediction(
        db_session,
        smiles="CCO",
        rf_prediction=-0.12,
        nn_prediction=-0.09,
        rf_model_version_id=None,
        nn_model_version_id=None,
        descriptors={"molecular_weight": 46.07},
        molecule_name="Ethanol",
        client_ip="127.0.0.1",
    )
    items, total = get_recent_predictions(db_session)
    assert total == 1
    assert items[0].id == row.id
    assert items[0].descriptors["molecular_weight"] == 46.07


def test_list_model_versions_returns_active_first(db_session) -> None:
    upsert_model_version(
        db_session,
        name="neural_network",
        version="2026.04.12-nn",
        artifact_path="models/neural_network.pt",
        mlflow_run_id="run-1",
        trained_at=None,
        cv_r2_mean=0.75,
        cv_rmse_mean=1.05,
        test_r2=0.74,
        test_rmse=1.08,
        hyperparameters={"lr": 0.001},
    )
    rows = list_model_versions(db_session, limit=10)
    assert rows[0].is_active is True
    assert isinstance(rows[0], ModelVersion)
