"""Pytest fixtures for SolPredict tests."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy.orm import Session, sessionmaker

from api.deps import get_db
from api.main import create_app
from solpredict.config import get_settings
from solpredict.db.engine import make_engine
from solpredict.db.models import Base
from solpredict.model import SolubilityMLP


@pytest.fixture(scope="session", autouse=True)
def _ensure_model_artifacts() -> None:
    """Provide minimal model artifacts for API tests when real ones are absent.

    Runs before any test imports `api.main` (which loads models in its lifespan).
    If real artifacts already exist, leaves them alone.
    """
    settings = get_settings()
    model_dir = Path(settings.model_dir)
    rf_path = model_dir / "random_forest.pkl"
    nn_path = model_dir / "neural_network.pt"

    if rf_path.exists() and nn_path.exists():
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_samples, n_bits = 32, 2048
    x = rng.integers(0, 2, size=(n_samples, n_bits)).astype(np.float32)
    y = rng.normal(size=n_samples).astype(np.float32)

    if not rf_path.exists():
        rf = RandomForestRegressor(n_estimators=5, random_state=0)
        rf.fit(x, y)
        joblib.dump(rf, rf_path)

    if not nn_path.exists():
        nn = SolubilityMLP(input_dim=n_bits)
        nn.eval()
        torch.save(nn.state_dict(), nn_path)


@pytest.fixture
def db_engine():
    engine = make_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def db_session(db_engine):
    with Session(db_engine, expire_on_commit=False) as session:
        yield session


@pytest.fixture
def db_session_factory(db_engine):
    return sessionmaker(
        bind=db_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


@pytest.fixture
def client(db_session_factory, monkeypatch):
    monkeypatch.setenv("SOLPREDICT_SKIP_MIGRATIONS", "1")
    monkeypatch.setattr("api.main.get_session_factory", lambda: db_session_factory)
    monkeypatch.setattr("api.deps.get_session_factory", lambda: db_session_factory)
    app = create_app()

    def override_get_db():
        db = db_session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
