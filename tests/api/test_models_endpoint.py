from solpredict.db.repositories import upsert_model_version


def test_models_returns_registered_versions(client, db_session) -> None:
    upsert_model_version(
        db_session,
        name="random_forest",
        version="2026.04.12-rf",
        artifact_path="models/random_forest.pkl",
        mlflow_run_id="run-123",
        trained_at=None,
        cv_r2_mean=0.71,
        cv_rmse_mean=1.17,
        test_r2=0.70,
        test_rmse=1.20,
        hyperparameters={"n_estimators": 100},
    )

    response = client.get("/models")
    assert response.status_code == 200
    rows = response.json()
    assert rows[0]["name"] == "random_forest"
    assert rows[0]["is_active"] is True
