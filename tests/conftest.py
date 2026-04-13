"""Pytest fixtures for SolPredict tests."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
from sklearn.ensemble import RandomForestRegressor

from solpredict.config import get_settings
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
