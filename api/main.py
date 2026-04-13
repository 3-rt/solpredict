"""FastAPI application factory for SolPredict."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import torch
from alembic.config import Config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from alembic import command
from api.routes.history import router as history_router
from api.routes.models import router as models_router
from api.routes.predict import router as predict_router
from solpredict.config import get_settings
from solpredict.db.engine import get_session_factory
from solpredict.db.repositories import get_active_model
from solpredict.model import SolubilityMLP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def run_startup_migrations() -> None:
    if os.getenv("SOLPREDICT_SKIP_MIGRATIONS") == "1":
        return

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    command.upgrade(config, "head")


def load_models(app: FastAPI) -> None:
    rf_model = None
    nn_model = None
    rf_model_version = None
    nn_model_version = None

    session_factory = get_session_factory()
    with session_factory() as session:
        rf_model_version = get_active_model(session, "random_forest")
        nn_model_version = get_active_model(session, "neural_network")

    rf_path = (
        Path(rf_model_version.artifact_path)
        if rf_model_version
        else MODEL_DIR / "random_forest.pkl"
    )
    nn_path = (
        Path(nn_model_version.artifact_path)
        if nn_model_version
        else MODEL_DIR / "neural_network.pt"
    )

    if rf_path.exists():
        rf_model = joblib.load(rf_path)
    if nn_path.exists():
        nn_params = nn_model_version.hyperparameters if nn_model_version else {}
        hidden_dims = nn_params.get("hidden_dims", (512, 128))
        if isinstance(hidden_dims, list):
            hidden_dims = tuple(hidden_dims)
        dropout = float(nn_params.get("dropout", 0.2))
        nn_model = SolubilityMLP(
            input_dim=get_settings().fp_nbits,
            hidden_dims=tuple(hidden_dims),
            dropout=dropout,
        )
        nn_model.load_state_dict(torch.load(nn_path, map_location="cpu", weights_only=True))
        nn_model.eval()

    app.state.rf_model = rf_model
    app.state.nn_model = nn_model
    app.state.rf_model_version = rf_model_version
    app.state.nn_model_version = nn_model_version


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_startup_migrations()
    load_models(app)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="SolPredict API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(predict_router)
    app.include_router(history_router)
    app.include_router(models_router)
    return app


app = create_app()
